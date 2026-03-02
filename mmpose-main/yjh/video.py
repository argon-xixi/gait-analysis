import moviepy
from moviepy.editor import *
video = VideoFileClip("/home/yjh/code_yjh/mmpose-main/vis_results/final.mp4")

print(f"video size: {video.size}") 
# #时间上裁剪
# video.subclip(25, 30).write_videofile("subclip_video.mp4")
#空间上裁剪
clip1=video.crop(x1=0,y1=0,x2=800,y2=980)
clip2=video.crop(x1=0,y1=980,x2=800,y2=1960)
min_height = min(clip1.size[1], clip2.size[1])
clip2 = clip2.set_position(("right", 0))
final_clip = CompositeVideoClip([clip1, clip2], 
                                   size=(clip1.w + clip2.w, min_height))
# final_clip=concatenate_videoclips([new_video1,new_video2.set_position((new_video1.size[0],0))])
final_clip.write_videofile('final_2.mp4',audio=False)