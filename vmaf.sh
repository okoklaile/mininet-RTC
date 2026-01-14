 vmaf-ffmpeg \
 -f rawvideo -pix_fmt yuv420p -r 60 -s:v 640x480 -i output/BBR_outvideo.yuv \
 -f rawvideo -pix_fmt yuv420p -s:v 640x480 -r 60 -i output/BBR_invideo.yuv \
 -filter_complex "libvmaf=model=path=vmaf_v0.6.1.json:feature=name=psnr|name=float_ms_ssim:framesync=ts_sync_mode=nearest" -f null -

 for offset in 0 0.016 0.033 0.05 0.1 0.2 0.5; do
  echo "Testing offset: $offset s"
   vmaf-ffmpeg -f rawvideo -pix_fmt yuv420p -s:v 640x480 -r 60 -ss $offset -i BBR_outvideo.yuv \
  -f rawvideo -pix_fmt yuv420p -s:v 640x480 -r 60 -i BBR_invideo.yuv \
  -filter_complex "ssim" -f null - 2>&1 | grep "All:"
done