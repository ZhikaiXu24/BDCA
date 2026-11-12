# Benchmark results

12 G1 Moves clips, 26,477 frames, and 7 controlled corruptions. Global strength 50.

Environment: python 3.12.4, numpy 1.26.4, scipy 1.15.3, Pillow 11.3.0.

Mean foot skating decreased in 12/12 clips. The median per-clip change was -13.6%, and the median per-clip mean world-space joint displacement was 0.46 mm.

## Real motion

| Clip | Frames | Foot skating, mm/s before → after | Mean penetration, mm before → after | Jerk P95 change | Mean joint displacement, mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| J_Dance0_StepTouch | 1949 | 36.7 → 29.1 | 0.27 → 0.24 | -10.9% | 0.46 |
| J_ShortDance16_JazzWalk | 637 | 59.0 → 47.4 | 0.45 → 0.40 | -7.6% | 0.83 |
| J_ShortDance14_Disco | 858 | 54.9 → 44.6 | 0.16 → 0.14 | -8.4% | 0.65 |
| J_Dance17_Shuffle | 1922 | 70.3 → 64.6 | 0.33 → 0.31 | -7.9% | 0.63 |
| B_SpiralDance | 2868 | 42.3 → 35.5 | 0.23 → 0.22 | -4.3% | 0.47 |
| B_DadDance | 2509 | 27.5 → 24.1 | 0.08 → 0.08 | -1.2% | 0.36 |
| J_Dance2_Salsa | 2151 | 78.6 → 67.7 | 0.27 → 0.25 | -4.1% | 0.81 |
| B_LongDance | 7167 | 67.1 → 62.3 | 0.39 → 0.35 | -7.1% | 0.67 |
| B_SpinKarate | 1540 | 42.5 → 37.9 | 0.38 → 0.34 | -6.8% | 0.29 |
| B_ForwardKarate | 2180 | 42.0 → 38.3 | 0.14 → 0.14 | -6.7% | 0.34 |
| B_BowKarate | 2254 | 28.6 → 24.8 | 0.22 → 0.21 | -1.2% | 0.28 |
| B_HandsUp | 442 | 8.6 → 6.6 | 0.00 → 0.00 | -0.5% | 0.09 |

## Controlled corruptions

RMS is measured against the clean reference within the injected interval. Only the module under test is enabled. The speed module uses an individual strength of 100.

| Corruption | Clean clip | RMS before → after | Change |
| --- | --- | ---: | ---: |
| elbow_jitter | J_Dance0_StepTouch | 1.12 → 0.73 deg | -35.2% |
| leg_jitter | B_SpinKarate | 1.12 → 0.35 deg | -69.2% |
| root_jitter | J_Dance2_Salsa | 4.31 → 1.93 mm | -55.3% |
| stance_slide | B_HandsUp | 26.63 → 5.79 mm | -78.3% |
| ground_sag | B_HandsUp | 8.12 → 7.86 mm | -3.2% |
| torso_spike | B_BowKarate | 23.00 → 21.88 deg | -4.9% |
| speed_spike | J_ShortDance16_JazzWalk | 95.00 → 78.80 deg | -17.1% |

## Strength sweep

| Clip | Strength | Foot skating change | Jerk P95 change | Mean joint displacement, mm |
| --- | ---: | ---: | ---: | ---: |
| J_Dance0_StepTouch.bvh | 0 | 0.0% | 0.0% | 0.00 |
| J_Dance0_StepTouch.bvh | 50 | -20.8% | -10.9% | 0.46 |
| J_Dance0_StepTouch.bvh | 100 | -21.9% | -9.1% | 0.53 |
| J_ShortDance16_JazzWalk.bvh | 0 | 0.0% | 0.0% | 0.00 |
| J_ShortDance16_JazzWalk.bvh | 50 | -19.7% | -7.6% | 0.83 |
| J_ShortDance16_JazzWalk.bvh | 100 | -23.4% | -5.7% | 0.96 |
| B_SpinKarate.bvh | 0 | 0.0% | 0.0% | 0.00 |
| B_SpinKarate.bvh | 50 | -10.9% | -6.8% | 0.29 |
| B_SpinKarate.bvh | 100 | -13.3% | -4.9% | 0.34 |

## Interpretation

- The input defines the contact intervals and ground estimate for both measurements. Foot skating uses horizontal heel/toe speed over consecutive support frames.
- Penetration uses virtual foot points and an estimated plane. Deep penetration and large torso spikes can exceed the bounded corrections.
- Jerk P95 describes high-end motion variation and can also reflect intentional fast motion.
- These are human-motion kinematics. Robot retargeting, simulation and hardware performance remain untested.
- G1 Moves by Experiential Technologies and the credited performers, CC BY 4.0. [Dataset revision](https://huggingface.co/datasets/exptech/g1-moves/tree/895d064b2385725e6aabf540461c219afc3e0916)
