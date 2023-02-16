import streamlit as st
from PIL import Image
import os
from itertools import cycle
import pandas as pd
import numpy as np
from os import listdir
from math import ceil
import faiss
import pickle

st.markdown("<h1 style='text-align: left; font-size: 27px; color: red;'>Movie character retrieval based on image examples</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align:left; font-size: 22px;'>Image Query</h1>",unsafe_allow_html=True)

#Left menu
st.sidebar.markdown("<h1 style='text-align:left; font-size: 20px;'>Select input image query</h1>",unsafe_allow_html=True)
st.sidebar.markdown("<h1 style='font-size: 16px;'>Choose movie</h1>", unsafe_allow_html=True)
lst_movies = ['Calloused Hands', 'Memphis', 'Liberty Kid', 'Losing Ground', 'Like Me']
option_movie = st.sidebar.selectbox('Choose movie', lst_movies, label_visibility="collapsed")


root_ground_truth ="ground_truth"
root_query ="query"
root_shots ="shots"
root_thumbnail ="thumbnail"
root_features = "features_5fps/deepface"
root_features_query = "features_query/deepface"
root_faiss_index = "faiss_index"

option_movie_dir=[]

if option_movie == 'Calloused Hands':
    lst_character = ['Byrd', 'Debbie']
    option_movie_dir = "Calloused_Hands"
if option_movie == 'Memphis':
    lst_character = ['Willis']
    option_movie_dir = "Memphis"
if option_movie == 'Liberty Kid':
    lst_character = ['Derrick']
    option_movie_dir = "Liberty_Kid"
if option_movie == 'Losing Ground':
    lst_character = ['Sara']
    option_movie_dir = "losing_ground"
if option_movie == 'Like Me':
    lst_character = ['Burt', 'Kiya']
    option_movie_dir = "like_me"

st.sidebar.markdown("<h1 style='font-size: 16px;'>Choose character</h1>", unsafe_allow_html=True)
option_character = st.sidebar.selectbox('Choose character',
                                       lst_character , label_visibility="collapsed")

feature_type = st.sidebar.radio("Face feature types", ["ArcFace", "FaceNet512"])
topk_value = st.sidebar.number_input('Choose top_k result', min_value=1, max_value=1000, step=1, value=5)
#batch_size = st.sidebar.select_slider("Batch size:",range(10,110,10))
batch_size = 10
column_size = st.sidebar.select_slider("Column size:", range(1,6), value = 2)

Images = []
path_to_imgs_query = os.path.join(root_query, option_movie_dir,option_character)
lst_images = sorted(os.listdir(path_to_imgs_query))

columns_imgs = st.columns(len(lst_images))
for file in lst_images:
        #if file.endswith(".png"):
        image = Image.open(os.path.join(path_to_imgs_query, file))
        image = image.resize((120,120))
        Images.append(image)
for idx, Image in enumerate(Images):
        #next(cols).image(Image)
        str_key = "chk_{}".format(idx)            
        with columns_imgs[idx]:
            st.image(Image)
            st.checkbox(lst_images[idx].split('.')[0], key=str_key)

#def submit_click(image):
#st.write(st.session_state)
#button = st.button("Submit", on_click=submit_click)
###################################################################################################

path_to_gt = os.path.join(root_ground_truth, option_movie_dir, "{}.xlsx".format(option_character))
path_to_file_index = os.path.join(root_faiss_index, feature_type, option_movie_dir, "{}.pickle".format(option_movie_dir))
path_to_file_index_fnames = os.path.join(root_faiss_index, feature_type, option_movie_dir, "{}-fnames.pickle".format(option_movie_dir))

#Load file index
index = faiss.read_index(path_to_file_index)
with open(path_to_file_index_fnames, 'rb') as f:
    lst_face_name = pickle.load(f)
list_fname = lst_face_name

#Load groundtruth
df = pd.read_excel(path_to_gt)
df_ground_truth_full = df["Full"]
lst_groundtruth = list(df_ground_truth_full)

#Chuyển thành ma trận và tạo danh sách kết quả
MAX_SCENES = 100
MAX_SHOTS = 200

lst_matrix_distance_min = []

def Refine_Result(lst_distance, lst_indices, list_fname):
    matrix_distance_min = -np.ones((MAX_SCENES, MAX_SHOTS))

    for dist, idx in zip(lst_distance, lst_indices):
        predicted_face = list_fname[idx]
        # print(predicted_face)
        ind = predicted_face.split('-')
        scene_id = int(ind[1])
        shot_id = int(ind[2].split('_')[1])
        # print(f"scene_id={scene_id}, shot_id={shot_id}")

        distance_min = matrix_distance_min[scene_id, shot_id]
        if distance_min == -1:
            distance_min = dist
        else:
            distance_min = min(dist, distance_min)

        matrix_distance_min[scene_id, shot_id] = distance_min
    return matrix_distance_min

def Create_List_Results(matrix_distance_min, option_movie_dir):
    lst_results = []
    idx_distance_min = np.argwhere(matrix_distance_min > -1)
    arr_distance = []
    for idx in idx_distance_min:
        arr_distance.append(matrix_distance_min[idx[0], idx[1]])
    arr_distance = np.array(arr_distance)
    idx_sorted = arr_distance.argsort()

    for jdx in idx_sorted:
        result = f'{option_movie_dir}-{idx_distance_min[jdx][0]}-shot_{idx_distance_min[jdx][1]}'
        lst_results.append(result)
    return lst_results

#Tạo danh sách kết quả cho face query 1 nếu được chọn
if 'chk_0' in st.session_state:
    if st.session_state.chk_0 == True:
        #st.write('Ảnh được chọn = ', lst_images[0])
        #Đọc feature query
        path_to_features_query = os.path.join(root_features_query, feature_type, option_movie_dir, option_character, "{}.pickle".format(lst_images[0].split('.')[0]))
        filename_feature_query = open(path_to_features_query, 'rb')
        feature_query = np.array(pickle.load(filename_feature_query))

        #Thực hiện truy vấn
        distance, indices = index.search(feature_query.reshape(1,-1), k=len(lst_face_name))

        #Chuyển thành ma trận và tạo danh sách kết quả
        matrix_distance_min = Refine_Result(distance[0],indices[0],list_fname)
        lst_results_0 = Create_List_Results(matrix_distance_min,option_movie_dir)
        lst_matrix_distance_min.append(matrix_distance_min)

#Tạo danh sách kết quả cho face query 2 nếu được chọn
if 'chk_1' in st.session_state:
    if st.session_state.chk_1 == True:
        # st.write('Ảnh được chọn = ', lst_images[1])
        #Đọc feature query
        path_to_features_query_1 = os.path.join(root_features_query, feature_type, option_movie_dir, option_character, "{}.pickle".format(lst_images[1].split('.')[0]))
        filename_feature_query_1 = open(path_to_features_query_1, 'rb')
        feature_query_1 = np.array(pickle.load(filename_feature_query_1))

        #Thực hiện truy vấn
        distance_1, indices_1 = index.search(feature_query_1.reshape(1,-1), k=len(lst_face_name))

        #Chuyển thành ma trận và tạo danh sách kết quả
        matrix_distance_min_1 = Refine_Result(distance_1[0], indices_1[0], list_fname)
        lst_results_1 = Create_List_Results(matrix_distance_min_1, option_movie_dir)
        lst_matrix_distance_min.append(matrix_distance_min_1)

#Tạo danh sách kết quả cho face query 3 nếu được chọn
if 'chk_2' in st.session_state:
    if st.session_state.chk_2 == True:
        # st.write('Ảnh được chọn = ', lst_images[2])
        #Đọc feature query
        path_to_features_query_2 = os.path.join(root_features_query, feature_type, option_movie_dir, option_character, "{}.pickle".format(lst_images[2].split('.')[0]))
        filename_feature_query_2 = open(path_to_features_query_2, 'rb')
        feature_query_2 = np.array(pickle.load(filename_feature_query_2))

        #Thực hiện truy vấn
        distance_2, indices_2 = index.search(feature_query_2.reshape(1, -1), k=len(lst_face_name))

        #Chuyển thành ma trận và tạo danh sách kết quả
        matrix_distance_min_2 = Refine_Result(distance_2[0], indices_2[0], list_fname)
        lst_results_2 = Create_List_Results(matrix_distance_min_2, option_movie_dir)
        lst_matrix_distance_min.append(matrix_distance_min_2)

#Tạo danh sách kết quả cho face query 4 nếu được chọn
if 'chk_3' in st.session_state:
    if st.session_state.chk_3 == True:
        # st.write('Ảnh được chọn = ', lst_images[3])
        # Đọc feature query
        path_to_features_query_3 = os.path.join(root_features_query, feature_type, option_movie_dir, option_character, "{}.pickle".format(lst_images[3].split('.')[0]))
        filename_feature_query_3 = open(path_to_features_query_3, 'rb')
        feature_query_3 = np.array(pickle.load(filename_feature_query_3))

        # Thực hiện truy vấn
        distance_3, indices_3 = index.search(feature_query_3.reshape(1, -1), k=len(lst_face_name))

        # Chuyển thành ma trận và tạo danh sách kết quả
        matrix_distance_min_3 = Refine_Result(distance_3[0], indices_3[0], list_fname)
        lst_results_3 = Create_List_Results(matrix_distance_min_3, option_movie_dir)
        lst_matrix_distance_min.append(matrix_distance_min_3)

#Tạo danh sách kết quả cho face query 5 nếu được chọn
if 'chk_4' in st.session_state:
    if st.session_state.chk_4 == True:
        # st.write('Ảnh được chọn = ', lst_images[4])
        #Đọc feature query
        path_to_features_query_4 = os.path.join(root_features_query,  feature_type, option_movie_dir, option_character, "{}.pickle".format(lst_images[4].split('.')[0]))
        filename_feature_query_4 = open(path_to_features_query_4, 'rb')
        feature_query_4 = np.array(pickle.load(filename_feature_query_4))

        #Thực hiện truy vấn
        distance_4, indices_4 = index.search(feature_query_4.reshape(1, -1), k=len(lst_face_name))

        #Chuyển thành ma trận và tạo danh sách kết quả
        matrix_distance_min_4 = Refine_Result(distance_4[0], indices_4[0], list_fname)
        lst_results_4 = Create_List_Results(matrix_distance_min_4, option_movie_dir)
        lst_matrix_distance_min.append(matrix_distance_min_4)

#Tạo danh sách kết quả cho face query 6 nếu được chọn
if 'chk_5' in st.session_state:
    if st.session_state.chk_5 == True:
        # st.write('Ảnh được chọn = ', lst_images[5])
        #Đọc feature query
        path_to_features_query_5 = os.path.join(root_features_query, feature_type, option_movie_dir, option_character, "{}.pickle".format(lst_images[5].split('.')[0]))
        if os.path.exists(path_to_features_query_5):

            filename_feature_query_5 = open(path_to_features_query_5, 'rb')
            feature_query_5 = np.array(pickle.load(filename_feature_query_5))

        #Thực hiện truy vấn
            distance_5, indices_5 = index.search(feature_query_5.reshape(1, -1), k=len(lst_face_name))

        #Chuyển thành ma trận và tạo danh sách kết quả
            matrix_distance_min_5 = Refine_Result(distance_5[0], indices_5[0], list_fname)
            lst_results_5 = Create_List_Results(matrix_distance_min_5, option_movie_dir)
            lst_matrix_distance_min.append(matrix_distance_min_5)

#Tạo danh sách kết quả cho face query 7 nếu được chọn
if 'chk_6' in st.session_state:
    if st.session_state.chk_6 == True:
        # st.write('Ảnh được chọn = ', lst_images[1])
        #Đọc feature query
        path_to_features_query_6 = os.path.join(root_features_query, feature_type, option_movie_dir, option_character, "{}.pickle".format(lst_images[6].split('.')[0]))
        filename_feature_query_6 = open(path_to_features_query_6, 'rb')
        feature_query_6 = np.array(pickle.load(filename_feature_query_6))

        #Thực hiện truy vấn
        distance_6, indices_6 = index.search(feature_query_6.reshape(1, -1), k=len(lst_face_name))

        #Chuyển thành ma trận và tạo danh sách kết quả
        matrix_distance_min_6 = Refine_Result(distance_6[0], indices_6[0], list_fname)
        lst_results_6 = Create_List_Results(matrix_distance_min_6, option_movie_dir)
        lst_matrix_distance_min.append(matrix_distance_min_6)

#Tạo danh sách kết quả cho nhiều face query
lst_lst_lst_results = []

if len(lst_matrix_distance_min) > 0:
    matrix_matrix_result = lst_matrix_distance_min[0]

    for j in range(1, len(lst_matrix_distance_min)):
        for r1 in range(MAX_SCENES):
            for c1 in range(MAX_SHOTS):
                if (matrix_matrix_result[r1][c1] == -1) and (lst_matrix_distance_min[j][r1][c1] == -1):
                    matrix_matrix_result[r1][c1] = -1
                if (matrix_matrix_result[r1][c1] != -1) and (lst_matrix_distance_min[j][r1][c1] != -1):
                    matrix_matrix_result[r1][c1] = min(lst_matrix_distance_min[j][r1][c1],matrix_matrix_result[r1][c1])
                if (matrix_matrix_result[r1][c1] == -1) and (lst_matrix_distance_min[j][r1][c1] != -1):
                    matrix_matrix_result[r1][c1] = lst_matrix_distance_min[j][r1][c1]
                if (matrix_matrix_result[r1][c1] != -1) and (lst_matrix_distance_min[j][r1][c1] == -1):
                    matrix_matrix_result[r1][c1] = matrix_matrix_result[r1][c1]

    lst_lst_lst_results = Create_List_Results(matrix_matrix_result,option_movie_dir)
##############################################################################
#Đánh giá kết quả

# Evaluation Metrics For Information Retrieval
def EvaluationMetrics(lst_predict, lst_groundtruth):
    num_gt = len(lst_groundtruth)
    num_correct = 0
    pos_predict = 0
    arr_Precision = []
    arr_Recall = []
    arr_F1 = []
    arr_AP = []
    myAP = 0
    arr_Reciprocal_Rank = []

    First_Reciprocal_Rank = -1
    for predict in lst_predict:
        pos_predict = pos_predict + 1
        # print(f"predict = {predict}")
        if predict in lst_groundtruth:
            # print(f"predict in lst_groundtruth = {predict}")
            num_correct += 1
            myAP = myAP + num_correct / pos_predict
            Reciprocal_Rank = 1.0 / pos_predict

            arr_Reciprocal_Rank.append(Reciprocal_Rank)
            if First_Reciprocal_Rank == -1:
                First_Reciprocal_Rank = Reciprocal_Rank
        else:
            arr_Reciprocal_Rank.append(0)

        Precision_k = num_correct / pos_predict
        arr_Precision.append(Precision_k)

        Recall_k = num_correct / num_gt
        arr_Recall.append(Recall_k)
        if Precision_k + Recall_k > 0:
            F1_k = 2 * Precision_k * Recall_k / (Precision_k + Recall_k)
        else:
            F1_k = 0

        arr_F1.append(F1_k)
        if num_correct > 0:
            arr_AP.append(myAP / num_correct)
        else:
            arr_AP.append(0)

    return First_Reciprocal_Rank, arr_Reciprocal_Rank, arr_Precision, arr_Recall, arr_F1, arr_AP

if len(lst_lst_lst_results) > 0:
    lst_predict = lst_lst_lst_results
    First_Reciprocal_Rank, arr_Reciprocal_Rank, arr_Precision, arr_Recall, arr_F1, arr_AP = EvaluationMetrics(lst_predict, lst_groundtruth)
    average_precision = ("AP_1=%5.2f" % arr_AP[0],
                     "AP_5=%5.2f" % arr_AP[4],
                     "AP_10=%5.2f" % arr_AP[9],
                     "AP_50=%5.2f"% arr_AP[49],
                     "AP_100=%5.2f"% arr_AP[99])
########################################################################################################3
#df_show_results = lst_lst_lst_results

topk = topk_value

if topk > len(lst_lst_lst_results):
    topk = len(lst_lst_lst_results)

df_results = lst_lst_lst_results[:topk]
df_show_results = df_results

def get_movie(shot_i):
    arr_str = shot_i.split('-')
    selected_movie = arr_str[0]
    selected_scene =  f'{arr_str[0]}-{arr_str[1]}'
    return selected_movie
def initialize():    
    df_show_results = lst_lst_lst_results
#    df_show_results = pd.DataFrame({'file':files,
#                    'incorrect':[False]*len(files),
#                    'label':['']*len(files)})
#    df.set_index('file', inplace=True)
    return df_show_results

if 'df_show_results' not in st.session_state:
    df_show_results = initialize()
    st.session_state.df_show_results = df_show_results
#else:
    #df_show_results = st.session_state.df_show_results 

num_batches = ceil(len(df_show_results)/batch_size)
 
if 'st_lst_result' not in st.session_state:
    st_lst_result = st.empty()

if 'st_video' not in st.session_state:
    st_video = st.empty()

if 'selected_shot' not in st.session_state:
    st.session_state.selected_shot=''
else:
    shot_i = st.session_state.selected_shot
    if len(shot_i)>0 and option_movie_dir==get_movie(shot_i):
        arr_str = shot_i.split('-')
        selected_movie = arr_str[0]
        selected_scene =  f'{arr_str[0]}-{arr_str[1]}'
        path_to_video_file = os.path.join(root_shots, selected_movie,selected_scene,"{}.webm".format(shot_i))
        if os.path.exists(path_to_video_file):
            video_file = open(path_to_video_file, 'rb')
            video_bytes = video_file.read()
            st_video.video(video_bytes)
            st_lst_result.markdown(f"Playing video: {shot_i}")
        else:
            st_lst_result.markdown(f"Not found video: {shot_i}")
    else:
        st_video = st.empty()

str_infor = "Top <b>{}</b> results of <b>{}</b> in <b>{}</b>".format(topk, option_character, option_movie)
str_eval = "Average Precision: "
if len(lst_lst_lst_results) > 0:
    st.write(str_eval, average_precision)

str_results = """
<table style="border: 1px solid #cc9966; width: 100%;" cellspacing="0" cellpadding="10px">
<tbody>
<tr>
<td style="border-bottom-color: #FC3; border-bottom-style: solid; border-bottom-width: 1px;" bgcolor="#ec6a00" height="20"><span style="padding-top: 10px; padding-bottom: 20px; font-family: Arial; font-size: 14px; font-style: normal; color: white; font-weight: bold;">
""" +str_infor + """</span></td></tr>""" 
st.markdown(str_results, unsafe_allow_html=True)

page = st.selectbox("Page", range(1,num_batches+1))

str_results = """
</tbody>
</table>
"""
st.markdown(str_results, unsafe_allow_html=True)

#@st.cache
def update (shot_i,selected_movie,selected_scene):
    st.session_state.selected_movie = selected_movie
    st.session_state.selected_scene = selected_scene
    st.session_state.selected_shot = shot_i

        
if len(df_show_results) > 0:
    batch = df_show_results[(page-1)*batch_size : page*batch_size]

    grid = st.columns(column_size)
    col_id = 0
    for shot_i in batch: # Calloused_Hands-1-shot_1
        with grid[col_id]:
            arr_str = shot_i.split('-')
            selected_movie = arr_str[0]
            selected_scene =  f'{arr_str[0]}-{arr_str[1]}'

            path_to_thumb = os.path.join(root_thumbnail,selected_movie, selected_scene, shot_i)
            if not os.path.exists(path_to_thumb):
                path_to_file_thumb = 'thumbnail\like_me\like_me-1\like_me-1-shot_1\like_me-1-shot_1-frame_87.jpg'
            else:
                lst_thumb = os.listdir(path_to_thumb)
                if len(lst_thumb)>0:
                    path_to_file_thumb = os.path.join(path_to_thumb,lst_thumb[0])
                else:
                    path_to_file_thumb = 'thumbnail\like_me\like_me-1\like_me-1-shot_1\like_me-1-shot_1-frame_87.jpg'
            st.image(path_to_file_thumb, caption=shot_i,width=192)
            st.button("Play video", key=f'{shot_i}',
                        on_click=update, args=(shot_i,selected_movie,selected_scene))
        col_id = (col_id + 1) % column_size
