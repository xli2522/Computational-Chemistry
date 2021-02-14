import imvideo as imv
import time

start_time = time.time()

imv.local.timelapse('water_npt_video.avi', 20, r'*\GitHub\Computational-Chemistry\spc216_frames')

print("--- %s seconds ---" % (time.time() - start_time))