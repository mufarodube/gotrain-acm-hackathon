**Best validation score: 0.6448 mIoU** (foreground IoU 0.385). It comes from trial t06: Unet with a resnet50 encoder, 384 px, BCE+Dice with `dice_weight` 1, lr at the default, 60 epochs and patience 15.

I stopped after 11 of the 16 trials. The last 5 trials (t07–t11) all failed to beat t06, which meets the stop rule of 3 failures in a row.

**Dataset:** m-NeonTree has 270 training and 94 validation images, with a 400×400 RGB canopy image plus a coarse 40×40 canopy-height layer and a 369-channel hyperspectral cube. Foreground (trees) is about 14% of pixels. The hyperspectral cube has more than 32 channels, so it is skipped automatically. As I read the log, the model sees only the RGB image and the height layer.

**Results:**

| Trial | Change | Val score |
|---|---|---|
| t01 | Baseline: Unet, resnet34, 256 px | 0.630 |
| t02 | 384 px | 0.633 |
| t03 | DeepLabV3Plus | 0.630 |
| t04 | resnet50 encoder | 0.641 |
| t05 | `dice_weight` 2 | 0.635 |
| t06 | resnet50 at 384 px | **0.645** |
| t07 | FPN with resnet50 | 0.643 |
| t08 | lr 1e-4 | 0.632 |
| t09 | resnet101 encoder | 0.641 |
| t10 | lr 1e-3 | 0.636 |
| t11 | Seed 1 rerun of t06 | 0.641 |

**What helped:** A resnet50 encoder gave about +1 point over resnet34, and 384 px added a little on top of that.

**What did not help:**
- DeepLabV3Plus gave no gain over Unet.
- FPN and resnet101 were no better than Unet with resnet50.
- Raising `dice_weight` to 2 did not help.
- Both lower and higher learning rates were worse than the default.

**Confirmation:** Rerunning the best config with a different seed gave 0.6406, a drop of 0.4 points. That is within noise, so the t06 score holds up, but I would call the real level about 0.64.

**Caveats:**
- Most configs land between 0.63 and 0.645, and differences under about 0.3 points are noise. The ranking among resnet50 variants (Unet, FPN, 256 or 384 px) is therefore not reliable.
- Validation scores peak around epochs 15–30 and then decline slightly, which suggests overfitting on 270 images.

**What I would try next:**
- Stronger regularisation, such as higher `weight_decay` or fewer epochs.
- `UnetPlusPlus` or `MAnet`.
- An ensemble of seeds.
- Getting the canopy-height layer used more effectively. I couldn't tell from the config options how it is used.
