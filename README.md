# DRACtrl-ComfyUI
a ComfyUI custom node for DRA-Ctrl
use hunyuani2v model for 9 i2i tasks
```
["canny","coloring","deblurring","depth","depth_pred","fill","sr","style_transfer","subject_driven"]
```
![微信截图_20250717133839](https://github.com/user-attachments/assets/a17bf8de-578e-4bec-8084-beeb0018a01f)

## Demo
- prompt
  - A vibrant young woman with rainbow glasses, yellow eyes, and colorful feather accessory against a bright yellow background
- source image
  - ![](https://github.com/user-attachments/assets/aba330e7-0a07-4bfe-be90-28533859bfd5)
- ouput
  - ![](https://github.com/user-attachments/assets/e1c66452-0740-4491-b341-842c0f323107)
  - ![video](https://github.com/user-attachments/assets/63671e6e-bca1-4527-bcd2-266f51b46cec)
## How to use
- download [hyi2v_transformer_mmgp.safetensors](https://pan.quark.cn/s/b327404283ea) put it in `ComfyUI/models/diffusion_models`
- 下载[hyi2v_transformer_mmgp.safetensors](https://pan.quark.cn/s/b327404283ea)把它放进`ComfyUI/models/diffusion_models`文件目录
- you can find example workflows in `DRACtrl-ComfyUI/example_workflows`
- 你可以在`DRACtrl-ComfyUI/example_workflows`目录下找到示例工作流

