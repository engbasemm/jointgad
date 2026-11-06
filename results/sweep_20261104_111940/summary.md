
### F1_full_as_trained: FULL LIST: each model as trained (+ codecs through the baseline)

| | BPP | PSNR | MS_SSIM | mAP | mIoU_all | mIoU_no_void |
|---|---|---|---|---|---|---|
| RES50_Baseline_no_compression | – | – | – | 0.1997 | 0.4919 | 0.4632 |
| Model_A_ChengAtten | 0.9747 | 22.5154 | 0.9215 | 0.1835 | 0.4103 | 0.3792 |
| Model_B_ChengAtten | 0.6815 | 21.9540 | 0.9104 | 0.1724 | 0.4202 | 0.3912 |
| Model_C_ChengAtten | 0.3128 | 22.0709 | 0.9117 | 0.1778 | 0.4149 | 0.3844 |
| Model_D_ChengAtten | 0.1308 | 19.5736 | 0.8300 | 0.1616 | 0.3808 | 0.3510 |
| HM HEVC QP31 | 0.7273 | 31.6257 | 0.9665 | 0.1836 | 0.4680 | 0.4387 |
| HM HEVC QP33 | 0.5668 | 30.3231 | 0.9560 | 0.1810 | 0.4545 | 0.4247 |
| HM HEVC QP36 | 0.3812 | 28.5374 | 0.9375 | 0.1678 | 0.4269 | 0.3962 |
| HM HEVC QP39 | 0.2470 | 26.8504 | 0.9150 | 0.1591 | 0.3976 | 0.3656 |
| HM HEVC QP42 | 0.1552 | 25.2225 | 0.8879 | 0.1204 | 0.3525 | 0.3175 |
| HM HEVC QP45 | 0.0924 | 23.6616 | 0.8540 | 0.0866 | 0.3001 | 0.2627 |
| JPEG2000 cl10 | 1.1980 | 27.6201 | 0.9248 | 0.1608 | 0.4274 | 0.3975 |
| JPEG2000 cl15 | 0.7987 | 25.8993 | 0.9024 | 0.1259 | 0.3722 | 0.3390 |
| JPEG2000 cl25 | 0.4788 | 24.1246 | 0.8701 | 0.0845 | 0.2982 | 0.2628 |
| JPEG2000 cl50 | 0.2393 | 22.3556 | 0.8255 | 0.0325 | 0.2129 | 0.1735 |
| JPEG2000 cl75 | 0.1597 | 21.5717 | 0.7928 | 0.0168 | 0.1762 | 0.1364 |
| JPEG2000 cl95 | 0.1262 | 21.1756 | 0.7743 | 0.0103 | 0.1647 | 0.1249 |

### F4_full_recal_s0: FULL LIST: BN re-estimation, seed 0 (main)

| | BPP | PSNR | MS_SSIM | mAP | mIoU_all | mIoU_no_void |
|---|---|---|---|---|---|---|
| RES50_Baseline_no_compression | – | – | – | 0.1997 | 0.4919 | 0.4632 |
| Model_A_ChengAtten | 0.7657 | 26.6308 | 0.9412 | 0.1972 | 0.4821 | 0.4530 |
| Model_B_ChengAtten | 0.5462 | 25.3529 | 0.9272 | 0.1957 | 0.4855 | 0.4563 |
| Model_C_ChengAtten | 0.2468 | 25.8709 | 0.9280 | 0.2010 | 0.4720 | 0.4408 |
| Model_D_ChengAtten | 0.1004 | 22.2559 | 0.8552 | 0.1801 | 0.4226 | 0.3940 |

### F5_full_recal_s1: FULL LIST: BN re-estimation, seed 1

| | BPP | PSNR | MS_SSIM | mAP | mIoU_all | mIoU_no_void |
|---|---|---|---|---|---|---|
| RES50_Baseline_no_compression | – | – | – | 0.1997 | 0.4919 | 0.4632 |
| Model_A_ChengAtten | 0.7677 | 26.6456 | 0.9413 | 0.2006 | 0.4820 | 0.4530 |
| Model_B_ChengAtten | 0.5481 | 25.3672 | 0.9272 | 0.1974 | 0.4855 | 0.4564 |
| Model_C_ChengAtten | 0.2479 | 25.8861 | 0.9281 | 0.1962 | 0.4710 | 0.4397 |
| Model_D_ChengAtten | 0.1009 | 22.2635 | 0.8554 | 0.1789 | 0.4215 | 0.3929 |

### F7_full_recal_all: FULL LIST: BN re-estimation for every encoder incl. baseline

| | BPP | PSNR | MS_SSIM | mAP | mIoU_all | mIoU_no_void |
|---|---|---|---|---|---|---|
| RES50_Baseline_no_compression | – | – | – | 0.1985 | 0.4911 | 0.4623 |
| Model_A_ChengAtten | 0.7657 | 26.6308 | 0.9412 | 0.1972 | 0.4821 | 0.4530 |
| Model_B_ChengAtten | 0.5462 | 25.3529 | 0.9272 | 0.1957 | 0.4855 | 0.4563 |
| Model_C_ChengAtten | 0.2468 | 25.8709 | 0.9280 | 0.2010 | 0.4720 | 0.4408 |
| Model_D_ChengAtten | 0.1004 | 22.2559 | 0.8552 | 0.1801 | 0.4226 | 0.3940 |

### FA1_full_sender: FULL LIST: heads on sender features f_enc(x)

| | BPP | PSNR | MS_SSIM | mAP | mIoU_all | mIoU_no_void |
|---|---|---|---|---|---|---|
| RES50_Baseline_no_compression | – | – | – | 0.1997 | 0.4919 | 0.4632 |
| Model_A_ChengAtten | 0.7500 | 26.4683 | 0.9398 | 0.2041 | 0.4967 | 0.4688 |
| Model_B_ChengAtten | 0.5266 | 25.1945 | 0.9271 | 0.1959 | 0.5070 | 0.4798 |
| Model_C_ChengAtten | 0.2413 | 25.6072 | 0.9261 | 0.2028 | 0.5065 | 0.4776 |
| Model_D_ChengAtten | 0.0972 | 21.9535 | 0.8576 | 0.2104 | 0.5078 | 0.4802 |

### mAP across runs

| model | F1_full_as_trained | F4_full_recal_s0 | F5_full_recal_s1 | F7_full_recal_all | FA1_full_sender |
|---|---|---|---|---|---|
| RES50_Baseline_no_compression | 0.1997 | 0.1997 | 0.1997 | 0.1985 | 0.1997 |
| Model_A_ChengAtten | 0.1835 | 0.1972 | 0.2006 | 0.1972 | 0.2041 |
| Model_B_ChengAtten | 0.1724 | 0.1957 | 0.1974 | 0.1957 | 0.1959 |
| Model_C_ChengAtten | 0.1778 | 0.2010 | 0.1962 | 0.2010 | 0.2028 |
| Model_D_ChengAtten | 0.1616 | 0.1801 | 0.1789 | 0.1801 | 0.2104 |

### mIoU_all across runs

| model | F1_full_as_trained | F4_full_recal_s0 | F5_full_recal_s1 | F7_full_recal_all | FA1_full_sender |
|---|---|---|---|---|---|
| RES50_Baseline_no_compression | 0.4919 | 0.4919 | 0.4919 | 0.4911 | 0.4919 |
| Model_A_ChengAtten | 0.4103 | 0.4821 | 0.4820 | 0.4821 | 0.4967 |
| Model_B_ChengAtten | 0.4202 | 0.4855 | 0.4855 | 0.4855 | 0.5070 |
| Model_C_ChengAtten | 0.4149 | 0.4720 | 0.4710 | 0.4720 | 0.5065 |
| Model_D_ChengAtten | 0.3808 | 0.4226 | 0.4215 | 0.4226 | 0.5078 |

### BPP across runs

| model | F1_full_as_trained | F4_full_recal_s0 | F5_full_recal_s1 | F7_full_recal_all | FA1_full_sender |
|---|---|---|---|---|---|
| RES50_Baseline_no_compression | – | – | – | – | – |
| Model_A_ChengAtten | 0.9747 | 0.7657 | 0.7677 | 0.7657 | 0.7500 |
| Model_B_ChengAtten | 0.6815 | 0.5462 | 0.5481 | 0.5462 | 0.5266 |
| Model_C_ChengAtten | 0.3128 | 0.2468 | 0.2479 | 0.2468 | 0.2413 |
| Model_D_ChengAtten | 0.1308 | 0.1004 | 0.1009 | 0.1004 | 0.0972 |
