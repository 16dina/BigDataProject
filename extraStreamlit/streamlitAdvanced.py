import streamlit as st
import os
from fastbook import *
from fastai.callback.fp16 import *
from fastai.vision.all import *
from PIL import Image


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
        with st.expander("# of images", expanded=False):
            dataset_train_dir = './datasetsingrid'
            classes_tr = os.listdir(dataset_train_dir)
            for class_tr in classes_tr:
                class_path = os.path.join(dataset_train_dir, class_tr)
                num_images = len(os.listdir(class_path))
                capitalized_class_tr = class_tr.capitalize()
                if "_" in capitalized_class_tr:
                    capitalized_class_tr = capitalized_class_tr.replace("_", " ")
                st.write(f"{capitalized_class_tr}: {num_images} images")

#selectbox with all the classes
with col2:
    dataset_dir = './datasetsingrid'
    categories = os.listdir(dataset_dir)
    displayed_categories = [category.capitalize().replace("_", " ") for category in categories]
    actual_categories = categories 
    selected_displayed_category = st.selectbox('Select a category', options=displayed_categories)
                  
with images:    
    #show 4 random images from each class
    def show_category_images(category_dir):
        images = os.listdir(category_dir)
        random.shuffle(images)  
        num_images_to_display = 4 

        col1, col2, col3, col4 = st.columns(4)

        for i in range(min(num_images_to_display, len(images))):
            image_path = os.path.join(category_dir, images[i])
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
        category_dir = os.path.join(dataset_dir, selected_category)
        images_placeholder = st.empty()
        show_category_images(category_dir)
        
with Model:
    col9, col10 = st.columns(2)
    col5, col6, col7, col8 = st.columns(4)
    

    with col9:
        st.header("Make your own model")
        col11, col12 = st.columns(2)
        with col11:
            #augmentations
            aug_map = {
                "Flip": Flip(),
                "Rotate": Rotate(),
                "Brightness": Brightness(),
                "Contrast": Contrast(),
                "Zoom": Zoom(),
            }
            aug_options = list(aug_map.keys())
            selected_aug = st.multiselect("Select Augmentations", aug_options)
            selected_tfms = [aug_map[t] for t in selected_aug]

            #epochs
            epochs = st.number_input("Number of Epochs", min_value=1, max_value=10, value=6)
        with col12:
            #learning rate
            learning_rate_options = {
                "0.1": 0.1,
                "0.01": 0.01,
                "0.001": 0.001,
                "0.00001": 0.00001,
                "0.000001": 0.000001,
            }
            lr_values = list(learning_rate_options.values())
            selected_lr = st.selectbox("Select Learning Rate", list(learning_rate_options.keys()))

            #architecture
            arch_options = {
                "ResNet34": resnet34,
                "ResNet50": resnet50,
                "Alexnet": alexnet,
                "VGG": vgg16_bn,
            }

            arch_names = list(arch_options.keys())
            selected_arch = st.selectbox("Select an Architecture", arch_names)


        if st.button("Train Model"):
            if len(selected_aug) > 0:
                #train the model
                leaves = DataBlock(
                    blocks=(ImageBlock, CategoryBlock),
                    get_items=get_image_files,
                    get_y=parent_label,
                    item_tfms=Resize(460),
                    batch_tfms=selected_tfms        
                )

                dls = leaves.dataloaders("./datasetsingrid")

                model = vision_learner(dls, arch_options[selected_arch], metrics=accuracy).to_fp16()
                model.fine_tune(epochs, freeze_epochs=3, base_lr=learning_rate_options[selected_lr])

                interp = ClassificationInterpretation.from_learner(model)
                st.write(interp.plot_confusion_matrix())

                class_names = model.dls.vocab                
            else:
                st.warning("Please select at least one transformation")

        
        st.header("Classify your image")

        #upload image
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    with col5:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
    with col5:
            #make predictions with the model
            classify_button = st.button("Classify")
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
    