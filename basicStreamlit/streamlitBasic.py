import streamlit as st
import os
from fastbook import *
from fastai.callback.fp16 import *
from fastai.vision.all import *
from PIL import Image
import pathlib


plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath


st.set_page_config(page_title="Leaf Classifier", page_icon="🍃", layout="wide")

header = st.container()
EDA = st.container()
images = st.container()
Model = st.container()
matrix = st.container()
with header:
    st.title("Leaf classifier")
with EDA:
    st.header("Performing exploratory data analysis")
    st.write("We wanted to work with the category leaves. We chose to scrape images of the following leaves: maple, birch, oak, fraxinus, ilex and salix fragilis.")
    st.write("We will start by showing the EDA that we did and a few sample images.")
    col2, col3, col4 = st.columns(3)

    with col2:
        #show how many images each class has
        with st.expander(":green[# of images]", expanded=False):
            dataset_train_dir = Path('./datasetsingrid')
            classes_tr = [entry.name for entry in dataset_train_dir.iterdir() if entry.is_dir()]

            #classes_tr = os.listdir(dataset_train_dir)
            for class_tr in classes_tr:
                class_path = dataset_train_dir / class_tr
                num_images = len(list(class_path.glob('*')))  # Counting images in the directory
                capitalized_class_tr = class_tr.capitalize().replace("_", " ")
                st.write(f"{capitalized_class_tr}: {num_images} images")

                # class_path = os.path.join(dataset_train_dir, class_tr)
                # num_images = len(os.listdir(class_path))
                # capitalized_class_tr = class_tr.capitalize()
                # if "_" in capitalized_class_tr:
                #     capitalized_class_tr = capitalized_class_tr.replace("_", " ")
                # st.write(f"{capitalized_class_tr}: {num_images} images")

#selectbox with all the classes
with col2:
    dataset_dir = Path('./datasetsingrid')
    categories = [entry.name for entry in dataset_dir.iterdir() if entry.is_dir()]
    displayed_categories = [category.capitalize().replace("_", " ") for category in categories]
    actual_categories = categories
    selected_displayed_category = st.selectbox('Select a category', options=displayed_categories)

    # dataset_dir = './datasetsingrid'
    # categories = os.listdir(dataset_dir)
    # displayed_categories = [category.capitalize().replace("_", " ") for category in categories]
    # actual_categories = categories 
    # selected_displayed_category = st.selectbox('Select a category', options=displayed_categories)
                  
with images:    
    #show 4 random images from each class
    def show_category_images(category_dir):
        images = list(Path(category_dir).glob('*'))
        random.shuffle(images)  
        num_images_to_display = 4 

        col1, col2, col3, col4 = st.columns(4)

        for i, image_path in enumerate(images[:num_images_to_display]):
            image = Image.open(image_path)
            if i % 4 == 0:
                col1.image(image, caption=f"Image {i + 1}")
            elif i % 4 == 1:
                col2.image(image, caption=f"Image {i + 1}")
            elif i % 4 == 2:
                col3.image(image, caption=f"Image {i + 1}")
            else:
                col4.image(image, caption=f"Image {i + 1}")
    
    if selected_displayed_category:
        selected_index = displayed_categories.index(selected_displayed_category)
        selected_category = actual_categories[selected_index]
        st.subheader(f"Images of category: {selected_displayed_category}")
        category_dir = dataset_dir / selected_category
        images_placeholder = st.empty()
        show_category_images(category_dir)

with Model:
    col9, col10 = st.columns(2)
    col5, col6, col7, col8 = st.columns(4)
    
    with col9:
        st.header("Classify your image")

        #model = load_learner('./export.pkl')
        model_path = Path('./export.pkl')
        model = load_learner(model_path)
        class_names = model.dls.vocab
        #upload images
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    with col5:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
    with col5:
            #make predictions with the model
            classify_button = st.button(":green[Classify]")
            if classify_button:
                with st.spinner('Classifying...'):
                    with col6:
                        prediction = model.predict(image)

                        st.write('Prediction:', prediction[0].capitalize())
                        st.write(" ")
                    
                        for idx, (class_name, probability) in enumerate(zip(class_names, prediction[2])):
                            if "_" in class_name:
                                class_name = class_name.replace("_", " ")
                            capitalized = class_name.capitalize()
                            st.write(f"{capitalized}: {probability.item()*100:.2f}%")


footer="""<style>

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}


</style>
<div class="footer">
<p>Ingrid Hansen, Dina Boshnaq and Iris Loret - Big Data Group 6</p>
</div>


"""
st.markdown(footer,unsafe_allow_html=True)
    