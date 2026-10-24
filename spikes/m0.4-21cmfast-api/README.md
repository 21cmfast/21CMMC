# M0.4: the v2 design against the 21cmFAST 4.3.1 API

Tracks [21cmfast/21CMMC#97](https://github.com/21cmfast/21CMMC/issues/97). I diffed the 21cmFAST 4.2 source (which the plan was written against) with 4.3.1, then probed the parts v2 relies on in a 4.3.1 build. `check_api.py` repeats the probes; run it before bumping the pin.

**Verdict: the design holds.** No structural change is needed. The plan needs four corrections and two new rules, listed below.

## Unchanged from 4.2

- `InputParameters.from_template(name, random_seed, node_redshifts, **kw)` and `evolve_input_structs(**kw)`. Unknown names still raise `TypeError`.
- `run_global_evolution(inputs, source_model, progressbar, overdensity_z0)`.
- `RectilinearLightconer.between_redshifts(...)` and lightconer `quantities`. A quantity that will not be computed raises `ValueError` before the run.
- `py21cmfast.wrapper.exceptions`: the same classes and hierarchy. `ParameterError` covers `GSLError`, `PhotonConsError`, `Table*Error`, `InfinityorNaNError` and `MassDepZetaError`. `FatalCError` covers `FileIOError`, `ArgumentValueError` and `MemoryAllocError`.
- `N_THREADS` still seeds the per-thread RNG streams (`src/rng.c` is identical) and is still part of the IC cache hash.

## Changed in 4.3, which v2 must follow

1. **Astro parameters were renamed** (the old names are deprecated, to be removed in 5.0):

   | Old | New |
   | --- | --- |
   | `F_STAR10`, `F_STAR7_MINI` | `F_STAR10_ACG`, `F_STAR7_MCG` |
   | `ALPHA_STAR`, `ALPHA_STAR_MINI` | `ALPHA_STAR_ACG`, `ALPHA_STAR_MCG` |
   | `F_ESC10`, `F_ESC7_MINI` | `F_ESC10_ACG`, `F_ESC7_MCG` |
   | `L_X`, `L_X_MINI` | `LX_OVER_SFR_ACG`, `LX_OVER_SFR_MCG` |
   | `M_TURN` | `M_TURN_STELLAR_FEEDBACK` |
   | `FIXED_VAVG` | `V_CB_AVG_DEBUG` |

   Option flags were renamed too: `USE_MINI_HALOS` → `USE_MCGS`, `FIX_VCB_AVG`/`USE_RELATIVE_VELOCITIES` → `V_CB_MODEL`, `INTEGRATION_METHOD_ATOMIC/MINI` → `_ACGS/_MCGS`. A new option, `USE_REIONIZATION_PHOTOHEATING_FEEDBACK`, was added.

   **A deprecated name cannot be sampled at all.** `evolve_input_structs(F_STAR10=x)` raises `ValueError` ("F_STAR10_ACG is set to -1.3 but F_STAR10 is x"). The reason is that `attrs.evolve` carries the new field's current value alongside the old one. The same happens the other way round: a base built from a TOML or kwargs using an old name refuses every later change to the new one. So v2's routing must use only the new names, and the 21cmFAST stage must reject a base that sets any deprecated alias.
2. **Output structs and fields were renamed:**
   - `HaloBox` → `EmissivityFields`, `XraySourceBox` → `RadiationFields`;
   - IonizedBox `log10_Mturnover_ave` → `log10_mturn_ave_acg`;
   - `CacheConfig(halobox=…)` → `emissivity_fields=…`;
   - config `EXTRA_HALOBOX_FIELDS` → `EXTRA_EMISSIVITY_FIELDS`.

   The lightcone quantities v2 uses are unaffected: `brightness_temp`, `neutral_fraction`, `ionisation_rate_G12`, `kinetic_temperature`, `density`, `spin_temperature`.
3. `generate_lightcone` lost its `cleanup` argument.
4. New: `config["CACHE_ARRAYS_ON_ACCESS"]` (default True) keeps arrays in memory once read from cache. Turn it off in workers to keep memory flat. There is also a new `py21cmfast.io.migrate` module: caches written by 4.2 are not found by 4.3 (the hashes changed), so they must be migrated or regenerated.

## Found while checking, not new in 4.3

- **Dependent defaults are frozen by `evolve_input_structs`.** These parameters are computed from other parameters when the struct is built:
  - `F_STAR7_MCG` (from `F_STAR10_ACG - 3·ALPHA_STAR_ACG`);
  - `ALPHA_STAR_MCG` and `LX_OVER_SFR_MCG` (from their ACG counterparts);
  - `X_RAY_Tvir_MIN` (from `ION_Tvir_MIN`).

  Evolving the source parameter does **not** update them. Example: evolving `F_STAR10_ACG` from the template to -1.0 leaves `F_STAR7_MCG` at -2.8; building fresh with -1.0 gives -2.5. This only matters with `USE_MCGS`, `X_RAY_Tvir_MIN`, or when the dependent parameter isn't sampled itself. But it changes the physics silently. 4.2 behaved the same way.
- **Invalid values raise a plain `ValueError`, not `ParameterError`.** This covers out-of-range values (e.g. `F_STAR10_ACG must be between -3.0 and 0.0`) and cross-struct checks (e.g. `R_BUBBLE_MAX > BOX_LEN/3`), all raised from attrs validators.
- The lightcone always adds `los_velocity` (used for `tau_21` when `include_dvdr_in_tau21=True`, the default), even if it wasn't requested. That is one extra lightcone-sized array.
- 4.3.1 has no wheels on PyPI and isn't on conda-forge yet. `pip install` builds from source and needs GSL and FFTW headers.

## Reproduce

```bash
python -m venv v431 && v431/bin/pip install 21cmFAST==4.3.1   # needs libgsl-dev, libfftw3-dev
v431/bin/python -W ignore check_api.py --run
```
