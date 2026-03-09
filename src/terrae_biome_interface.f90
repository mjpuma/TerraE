! =============================================================================
! terrae_biome_interface.f90
!
! Thin adapter module providing the TerraE ↔ BiomeE coupling interface.
!
! DESIGN PRINCIPLE:  Minimal changes to existing BiomeE source code.
!   - Add ONE line  "USE terrae_biome_interface"  to BiomeE's soil_water
!     subroutine (and optionally to physiology.f90 for canopy conductance).
!   - Replace BiomeE's bucket soil update with a call to biome_set_soil_from_terrae().
!   - All other BiomeE code (photosynthesis, demography, C/N cycles) unchanged.
!
! INTERFACE SUBROUTINES (called by TerraE Python via f2py or cffi):
!   biome_set_forcings  – TerraE pushes atmospheric + soil state to BiomeE
!   biome_get_exports   – TerraE pulls canopy properties + transpiration from BiomeE
!   biome_run           – advance BiomeE one GCM timestep (or no-op if loose coupling)
!
! SUBROUTINES CALLED BY BIOMEE (to push soil state and receive exports):
!   biome_set_soil_from_terrae – BiomeE calls with its arrays; interface fills them
!   biome_receive_exports      – BiomeE calls to push canopy outputs into interface
!
! OPEN ISSUE [FLAG: PSI_STRESS]:
!   BiomeE's water stress beta function currently uses theta relative to
!   field capacity (theta_fc) and wilting point (theta_wp).  A future
!   improvement is to replace this with a matric-potential-based beta(psi),
!   which is more physically consistent and directly uses TerraE's psi_k.
!   The interface already passes psi_k to BiomeE; the stress function update
!   requires changes inside BiomeE's physiology subroutine and is deferred.
!   See Section 9.7 of the TerraE Technical Description.
!
! COMPILATION (gfortran, standalone test):
!   gfortran -c terrae_biome_interface.f90 -o terrae_biome_interface.o
!   # Then link with BiomeE objects and terrae Python extension (f2py/cffi).
!
! AUTHOR:  TerraE development team
! DATE:    2025
! =============================================================================

MODULE terrae_biome_interface

  IMPLICIT NONE

  ! ── Public API ───────────────────────────────────────────────────────────
  PUBLIC :: biome_set_forcings, biome_get_exports, biome_run
  PUBLIC :: biome_set_soil_from_terrae, biome_receive_exports, init_interface
  PUBLIC :: biome_set_temp_unit, NSOIL, NBIOME

  ! ── Dimensions (match BiomeE and TerraE configuration) ─────────────────────
  INTEGER, PARAMETER :: NSOIL = 10   ! max soil layers (TerraE NGM; BiomeE nsoil)
  INTEGER, PARAMETER :: NBIOME = 6   ! max BiomeE land cover columns per grid cell

  ! ── Storage: soil state received FROM TerraE, one column ───────────────────
  ! Indexed (layer k, column j); TerraE fills these before biome_run() is called.
  REAL(8), SAVE :: te_theta(NSOIL, NBIOME)   ! volumetric water content [m3/m3]
  REAL(8), SAVE :: te_Tsoil(NSOIL, NBIOME)   ! soil temperature [K]
  REAL(8), SAVE :: te_psi  (NSOIL, NBIOME)   ! matric potential [m H2O; negative unsaturated]
  REAL(8), SAVE :: te_dz   (NSOIL)           ! layer thicknesses [m]
  INTEGER, SAVE :: te_nsoil                  ! number of active layers
  INTEGER, SAVE :: te_ncols                  ! number of active land-cover columns

  ! ── Storage: atmospheric forcings received FROM TerraE ─────────────────────
  REAL(8), SAVE :: te_precip    ! precipitation [m/s]
  REAL(8), SAVE :: te_srht      ! shortwave radiation [W/m2]
  REAL(8), SAVE :: te_trht      ! longwave radiation [W/m2]
  REAL(8), SAVE :: te_Tair      ! surface air temperature [K]
  REAL(8), SAVE :: te_qs        ! specific humidity [kg/kg]
  REAL(8), SAVE :: te_vs        ! wind speed [m/s]
  REAL(8), SAVE :: te_pres      ! surface pressure [Pa]
  REAL(8), SAVE :: te_dt        ! GCM timestep [s]

  ! ── Storage: canopy exports TO TerraE, one column ─────────────────────────
  REAL(8), SAVE :: bm_cc     (NBIOME)        ! canopy conductance [m/s]
  REAL(8), SAVE :: bm_lai    (NBIOME)        ! leaf area index [m2/m2]
  REAL(8), SAVE :: bm_sai    (NBIOME)        ! stem area index [m2/m2]
  REAL(8), SAVE :: bm_vh     (NBIOME)        ! vegetation height [m]
  REAL(8), SAVE :: bm_transp (NBIOME)        ! actual whole-column transpiration [m/s]
  REAL(8), SAVE :: bm_froot  (NSOIL, NBIOME) ! root fraction per layer [-], sums to 1
  REAL(8), SAVE :: bm_fr_snow_c(NBIOME)      ! fraction of canopy covered by snow [-]
  REAL(8), SAVE :: bm_clump  (NBIOME)        ! canopy clumping factor [-]

  ! ── Initialization flag ────────────────────────────────────────────────────
  LOGICAL, SAVE :: initialized = .FALSE.

  ! ── Optional: temperature unit for BiomeE (K vs °C) ───────────────────────────
  ! Set .TRUE. if BiomeE expects Celsius; .FALSE. if Kelvin
  LOGICAL, SAVE :: biome_uses_celsius = .TRUE.

CONTAINS

  ! ===========================================================================
  ! biome_set_forcings
  !   Called by TerraE (Python side) once per GCM timestep BEFORE biome_run().
  !   Pushes atmospheric forcings and the time-averaged soil state (theta, T, psi)
  !   accumulated over TerraE's sub-steps into module-level storage.
  ! ===========================================================================
  SUBROUTINE biome_set_forcings(                            &
      precip, srht, trht, Tair, qs, vs, pres, dt,          &
      theta, Tsoil, psi, dz, nsoil, ncols)

    REAL(8), INTENT(IN) :: precip, srht, trht, Tair, qs, vs, pres, dt
    INTEGER, INTENT(IN) :: nsoil, ncols
    REAL(8), INTENT(IN) :: theta (nsoil, ncols)
    REAL(8), INTENT(IN) :: Tsoil (nsoil, ncols)
    REAL(8), INTENT(IN) :: psi   (nsoil, ncols)   ! matric potential [m]; <= 0
    REAL(8), INTENT(IN) :: dz    (nsoil)

    INTEGER :: k, j

    ! Bounds check to prevent buffer overflow
    IF (nsoil > NSOIL .OR. ncols > NBIOME) THEN
      WRITE(*,*) "biome_set_forcings: ERROR — nsoil=", nsoil, " or ncols=", ncols, &
          " exceeds NSOIL=", NSOIL, " NBIOME=", NBIOME
      STOP 1
    END IF

    te_precip = precip
    te_srht   = srht
    te_trht   = trht
    te_Tair   = Tair
    te_qs     = qs
    te_vs     = vs
    te_pres   = pres
    te_dt     = dt
    te_nsoil  = nsoil
    te_ncols  = ncols

    DO k = 1, nsoil
      te_dz(k) = dz(k)
      DO j = 1, ncols
        te_theta(k,j) = theta(k,j)
        te_Tsoil(k,j) = Tsoil(k,j)
        te_psi  (k,j) = psi  (k,j)
      END DO
    END DO

    initialized = .TRUE.

  END SUBROUTINE biome_set_forcings


  ! ===========================================================================
  ! biome_get_exports
  !   Called by TerraE (Python side) AFTER biome_run() each GCM timestep.
  !   Returns canopy properties and transpiration demand for each column.
  !
  !   TerraE uses these as:
  !     evapdl[k] = bm_froot[k,j] * bm_transp[j] / dz[k]   (transpiration sink)
  !     chvg computed from bm_cc, bm_lai, bm_sai, bm_clump  (soil evap)
  !     f_sn computed from bm_vh, bm_fr_snow_c               (snow masking)
  ! ===========================================================================
  SUBROUTINE biome_get_exports(                             &
      cc, lai, sai, vh, transp, froot, fr_snow_c, clump,   &
      nsoil, ncols)

    INTEGER, INTENT(IN)  :: nsoil, ncols
    REAL(8), INTENT(OUT) :: cc        (ncols)
    REAL(8), INTENT(OUT) :: lai       (ncols)
    REAL(8), INTENT(OUT) :: sai       (ncols)
    REAL(8), INTENT(OUT) :: vh        (ncols)
    REAL(8), INTENT(OUT) :: transp    (ncols)
    REAL(8), INTENT(OUT) :: froot     (nsoil, ncols)
    REAL(8), INTENT(OUT) :: fr_snow_c (ncols)
    REAL(8), INTENT(OUT) :: clump     (ncols)

    INTEGER :: k, j

    ! Bounds check
    IF (nsoil > NSOIL .OR. ncols > NBIOME) THEN
      WRITE(*,*) "biome_get_exports: ERROR — nsoil or ncols exceeds limits"
      STOP 1
    END IF

    DO j = 1, ncols
      cc       (j)   = bm_cc     (j)
      lai      (j)   = bm_lai    (j)
      sai      (j)   = bm_sai    (j)
      vh       (j)   = bm_vh     (j)
      transp   (j)   = bm_transp (j)
      fr_snow_c(j)   = bm_fr_snow_c(j)
      clump    (j)   = bm_clump  (j)
      DO k = 1, nsoil
        froot(k,j) = bm_froot(k,j)
      END DO
    END DO

  END SUBROUTINE biome_get_exports


  ! ===========================================================================
  ! biome_receive_exports
  !   Called BY BiomeE after its timestep to push canopy outputs into the
  !   interface. TerraE then retrieves them via biome_get_exports().
  !
  !   BiomeE integration: call this from BiomeE's output/export section
  !   after computing cc, lai, sai, transp, etc. for each column.
  ! ===========================================================================
  SUBROUTINE biome_receive_exports(                        &
      cc, lai, sai, vh, transp, froot, fr_snow_c, clump,  &
      ncols)

    INTEGER, INTENT(IN) :: ncols
    REAL(8), INTENT(IN) :: cc        (ncols)
    REAL(8), INTENT(IN) :: lai       (ncols)
    REAL(8), INTENT(IN) :: sai       (ncols)
    REAL(8), INTENT(IN) :: vh        (ncols)
    REAL(8), INTENT(IN) :: transp    (ncols)
    REAL(8), INTENT(IN) :: froot     (:,:)   ! (nsoil, ncols); min(nsoil)=1
    REAL(8), INTENT(IN) :: fr_snow_c (ncols)
    REAL(8), INTENT(IN) :: clump     (ncols)

    INTEGER :: k, j, nf

    IF (ncols > NBIOME) THEN
      WRITE(*,*) "biome_receive_exports: ERROR — ncols exceeds NBIOME"
      STOP 1
    END IF

    nf = MIN(SIZE(froot, 1), NSOIL)

    DO j = 1, ncols
      bm_cc     (j) = cc       (j)
      bm_lai    (j) = lai      (j)
      bm_sai    (j) = sai      (j)
      bm_vh     (j) = vh       (j)
      bm_transp (j) = transp   (j)
      bm_fr_snow_c(j) = fr_snow_c(j)
      bm_clump  (j) = clump    (j)
      DO k = 1, nf
        bm_froot(k,j) = froot(k,j)
      END DO
      ! Zero any extra layers if BiomeE has fewer than NSOIL
      DO k = nf + 1, NSOIL
        bm_froot(k,j) = 0.0d0
      END DO
    END DO

  END SUBROUTINE biome_receive_exports


  ! ===========================================================================
  ! biome_run
  !   Called by TerraE (Python side) once per GCM timestep.
  !   Advances BiomeE's biological processes (photosynthesis, growth, demography,
  !   C/N cycles) for one GCM timestep, using the soil state in te_theta/te_Tsoil.
  !   Does NOT advance soil water — that is TerraE's job.
  !
  !   TIGHT COUPLING: BiomeE is linked; it calls biome_set_soil_from_terrae(j)
  !   and biome_receive_exports from within its own timestep. This routine just
  !   invokes the BiomeE driver (if registered) or is a no-op.
  !
  !   LOOSE COUPLING: TerraE calls biome_set_forcings, then an external driver
  !   calls BiomeE separately, then BiomeE calls biome_receive_exports, then
  !   TerraE calls biome_get_exports. In that case biome_run can be a no-op.
  ! ===========================================================================
  SUBROUTINE biome_run(ncols)

    INTEGER, INTENT(IN) :: ncols
    INTEGER :: j

    IF (.NOT. initialized) THEN
      WRITE(*,*) "biome_run: ERROR — biome_set_forcings must be called first"
      STOP 1
    END IF

    IF (ncols > NBIOME) THEN
      WRITE(*,*) "biome_run: ERROR — ncols exceeds NBIOME"
      STOP 1
    END IF

    ! For each column: BiomeE (when linked) will call biome_set_soil_from_terrae(j)
    ! from within its soil_water routine. The actual BiomeE timestep is driven
    ! by BiomeE's own entry point (e.g. BiomeE_step). This stub documents the
    ! expected flow; replace with actual BiomeE driver call when linking:
    !   CALL BiomeE_step(te_dt)
    DO j = 1, ncols
      ! Placeholder: when tightly coupled, BiomeE's soil_water calls
      ! biome_set_soil_from_terrae(j, wcl, theta_soil, tsoil, psi_soil)
      ! and its output section calls biome_receive_exports(...)
      CONTINUE
    END DO

  END SUBROUTINE biome_run


  ! ===========================================================================
  ! biome_set_soil_from_terrae
  !
  !   Called BY BiomeE from its soil_water subroutine. BiomeE passes its
  !   arrays; this routine fills them from TerraE's te_theta, te_Tsoil, te_psi.
  !   No dependency on BiomeE module — BiomeE provides the arrays.
  !
  !   Usage in BiomeE's soil_water:
  !     USE terrae_biome_interface, ONLY: biome_set_soil_from_terrae
  !     ...
  !     CALL biome_set_soil_from_terrae(j, wcl=wcl, theta_soil=theta_soil, &
  !         tsoil=tsoil, psi_soil=psi_soil)
  !
  !   Optional arguments: pass only the arrays BiomeE uses. psi_soil is for
  !   future psi-based stress (Section 9.7).
  !
  !   Units: wcl [m], theta_soil [m3/m3], tsoil [°C or K per biome_uses_celsius],
  !         psi_soil [m H2O]
  ! ===========================================================================
  SUBROUTINE biome_set_soil_from_terrae(j, wcl, theta_soil, tsoil, psi_soil)

    INTEGER, INTENT(IN) :: j   ! column index (1..te_ncols)
    REAL(8), INTENT(OUT), OPTIONAL :: wcl(:)        ! water content depth [m]
    REAL(8), INTENT(OUT), OPTIONAL :: theta_soil(:) ! volumetric [m3/m3]
    REAL(8), INTENT(OUT), OPTIONAL :: tsoil(:)     ! temperature [°C or K]
    REAL(8), INTENT(OUT), OPTIONAL :: psi_soil(:)  ! matric potential [m H2O]

    INTEGER :: k
    REAL(8) :: Tk

    IF (j < 1 .OR. j > te_ncols) THEN
      WRITE(*,*) "biome_set_soil_from_terrae: ERROR — column j=", j, " out of range"
      STOP 1
    END IF

    DO k = 1, te_nsoil
      IF (PRESENT(wcl)) THEN
        wcl(k) = te_theta(k,j) * te_dz(k)
      END IF
      IF (PRESENT(theta_soil)) THEN
        theta_soil(k) = te_theta(k,j)
      END IF
      IF (PRESENT(tsoil)) THEN
        Tk = te_Tsoil(k,j)
        IF (biome_uses_celsius) THEN
          tsoil(k) = Tk - 273.15d0
        ELSE
          tsoil(k) = Tk
        END IF
      END IF
      IF (PRESENT(psi_soil)) THEN
        psi_soil(k) = te_psi(k,j)
      END IF
    END DO

  END SUBROUTINE biome_set_soil_from_terrae


  ! ===========================================================================
  ! biome_set_temp_unit
  !   Set whether BiomeE expects soil temperature in Celsius (.TRUE.) or Kelvin.
  !   Default is .TRUE. (Celsius). Call before first biome_set_forcings if needed.
  ! ===========================================================================
  SUBROUTINE biome_set_temp_unit(use_celsius)
    LOGICAL, INTENT(IN) :: use_celsius
    biome_uses_celsius = use_celsius
  END SUBROUTINE biome_set_temp_unit


  ! ===========================================================================
  ! init_interface
  !   Zero all module-level arrays. Call once at model startup.
  ! ===========================================================================
  SUBROUTINE init_interface()
    te_theta    = 0.0d0
    te_Tsoil    = 273.15d0
    te_psi      = -1.0d0   ! mild unsaturated initial state
    te_dz       = 0.0d0
    bm_cc       = 0.0d0
    bm_lai      = 0.0d0
    bm_sai      = 0.0d0
    bm_vh       = 0.0d0
    bm_transp   = 0.0d0
    bm_froot    = 0.0d0
    bm_fr_snow_c= 0.0d0
    bm_clump    = 1.0d0    ! default: no clumping
    te_nsoil    = 0
    te_ncols    = 0
    initialized = .FALSE.
  END SUBROUTINE init_interface

END MODULE terrae_biome_interface
