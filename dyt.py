from pytube import YouTube

# URL of the YouTube video
video_url = 'https://www.youtube.com/watch?v=191PshRLtos&ab_channel=JREClips'  # replace with your URL

# Create a YouTube object
yt = YouTube(video_url)

# Select the highest quality stream of the video
stream = yt.streams.get_highest_resolution()

# Download the video
stream.download(output_path=r'C:\Users\struc\Documents\GitHub\RTDAI\videos')  # specify the output path if needed
