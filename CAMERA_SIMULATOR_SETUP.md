# Camera Simulator Setup

## 1. Install Vimba X SDK

Install the Allied Vision Vimba X SDK.

---

## 2. Configure Simulator XML

Open

```
VimbaCameraSimulatorTL.xml
```

Set

```xml
<CustomImagesPath>

/home/xisai/Downloads/bmp_frames

</CustomImagesPath>
```

where `bmp_frames` contains the converted BMP images.

---

## 3. Configure GenTL

Run

```bash
cd ~/Downloads/VimbaX_2026-2/cti

source Set_GenTL_Path.sh
```

Verify

```bash
echo $GENICAM_GENTL64_PATH
```

---

## 4. Launch Live Inference

```bash
python live_inference.py
```

The simulator streams images as a virtual camera while the detection pipeline performs live inference.
