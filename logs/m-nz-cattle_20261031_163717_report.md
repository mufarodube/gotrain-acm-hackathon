## Report: m-nz-cattle

**Best config:** Unet with a resnet34 encoder, `img_size` 640, `lr` 1e-3, `dice_weight` 1, 30 epochs. The time budget (10 min) ended the run at about epoch 27.
- **Validation score:** 0.8517 (mean IoU), with foreground IoU of 0.711.
- **Seed check:** Rerunning with a different seed gave 0.8508, a drop of 0.1 points. That is within noise, so the result looks real.

**Data:** 524 training and 66 validation images, 500×500 RGB. Cattle cover only about 2% of pixels, so foreground IoU is the number to watch.

**What helped**
- **Higher learning rate:** `lr` 1e-3 beat the default, 0.8471 against 0.8400 (+0.7 points, a single run).
- **Larger `img_size`:** the score rose with image size. At lr 1e-3, 320 gave 0.8305, 480 gave 0.8471 and 640 gave 0.8517. Upsampling past the native 500 pixels helps because the objects are so small.

**What did not help (all within about 0.5 points of the best, or worse)**

| Change | Score |
|---|---|
| DeepLabV3Plus | 0.8251 |
| resnet50 encoder | 0.8413 |
| resnet18 encoder | 0.8470 |
| `dice_weight` 2 | 0.8475 |
| `batch_size` 4 | 0.8466 |
| `weight_decay` 0.01 | 0.8463 |

The last four are not clearly different from the 480 resolution baseline.

**Stopping:** The last five trials (t07–t11) did not beat the best, so I stopped with 5 trials unused. The 640 run was cut off by the time budget, but its validation curve had already flattened, so longer training is unlikely to add much.

**Caveats:** The validation set is only 66 images, so differences under about 0.5 points are not reliable. The 640 gain over 480 is about 0.5 points on a single seed.

**Next steps if continuing:**
- Try 800 pixels, which would need a longer time budget.
- Try a higher-capacity decoder or an Unet++ variant.
- Ensemble several seeds, or use test-time augmentation with flips and rotations.
- Try a focal or Tversky loss.
