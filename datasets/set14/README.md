# Set14 - third evaluation set

Held-out evaluation set added on 2026-10-30 to test whether the quantization conclusions transfer
beyond Kodak and CLIC. It was never used for calibration, coefficient fitting, or tuning.

## Contents

14 natural colour images at their native resolutions, 250x360 to 768x512. `img_003.png` is
grayscale; the rest are RGB.

| File | Size | Source name |
|---|---|---|
| `img_001.png` | 500x480 | `img_001_SRF_2_HR.png` |
| `img_002.png` | 720x576 | `img_002_SRF_2_HR.png` |
| `img_003.png` | 512x512 (grayscale) | `img_003_SRF_2_HR.png` |
| `img_004.png` | 352x288 | `img_004_SRF_2_HR.png` |
| `img_005.png` | 250x360 | `img_005_SRF_2_HR.png` |
| `img_006.png` | 276x276 | `img_006_SRF_2_HR.png` |
| `img_007.png` | 500x362 | `img_007_SRF_2_HR.png` |
| `img_008.png` | 352x288 | `img_008_SRF_2_HR.png` |
| `img_009.png` | 512x512 | `img_009_SRF_2_HR.png` |
| `img_010.png` | 512x512 | `img_010_SRF_2_HR.png` |
| `img_011.png` | 768x512 | `img_011_SRF_2_HR.png` |
| `img_012.png` | 512x512 | `img_012_SRF_2_HR.png` |
| `img_013.png` | 528x656 | `img_013_SRF_2_HR.png` |
| `img_014.png` | 586x390 | `img_014_SRF_2_HR.png` |

## Provenance

- Upstream: <https://github.com/jbhuang0604/SelfExSR>, directory `data/Set14/image_SRF_2`.
- The HR files are the original Set14 images; the `LR` and `SRF_3`/`SRF_4` variants are not used.
- Downloaded 2026-10-30. Per-file SHA-256 in `manifest.json`.
- File names were normalised from `img_NNN_SRF_2_HR.png` to `img_NNN.png`; bytes are unmodified.

## Why it is gitignored

`datasets/set14/*.png` is ignored for the same reason as `datasets/kodak/*.png` and
`datasets/clic_professional/`: the images are third-party, re-downloadable, and would add 5 MB to
every clone. Verify the upstream licence yourself before redistributing them. This README and
`manifest.json` are committed so the set can be restored and checked.

Re-download with:

```powershell
.\venv\Scripts\python.exe -c "import json,ssl,urllib.request,pathlib; ctx=ssl._create_unverified_context(); api='https://api.github.com/repos/jbhuang0604/SelfExSR/contents/data/Set14/image_SRF_2'; out=pathlib.Path('datasets/set14'); out.mkdir(parents=True,exist_ok=True); [ (out/it['name'].replace('_SRF_2_HR','')).write_bytes(urllib.request.urlopen(urllib.request.Request(it['download_url'],headers={'User-Agent':'pulse-quant/1.0'}),timeout=60,context=ctx).read()) for it in json.loads(urllib.request.urlopen(urllib.request.Request(api,headers={'User-Agent':'pulse-quant/1.0'}),timeout=30,context=ctx).read()) if it['name'].endswith('_HR.png') ]"
```

## Reference

The FP32 reference and noise floor live outside git, as for the other datasets:

```powershell
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\pipeline.py reference --dataset set14 --qps 0 1 2 3 4 5 6 7
```

Recorded noise floor: 112 image/QP pairs, TF32 off is bit-identical to the main pass
(`d_psnr` 0.0, `d_bpp` 0.0%, stream and reconstruction hashes identical). TF32 on moves
`d_psnr` by up to 0.0144 dB and `d_bpp` by up to 0.248%, which is why TF32 stays off.

## Evaluation

```powershell
.\venv\Scripts\python.exe quantization-pluse-mse\scripts\eval_assignment.py --dataset set14 `
  --configs fp32 fp16 bf16 int8 safe greedy ilp
```
