import os


################################################ Init Phase ####################################################################
path = 'images_original'
l_file = []
l_song_img = []

for elt in os.listdir('images_original'):
    l_file.append(os.listdir(path+'/'+elt))

for file in l_file:
    for song in file:
        l_song_img.append(path+'/'+song)

print(len(l_song_img))