import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import requests
import os
import time
from io import BytesIO


# Import torchvision
import torchvision

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# Preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.CenterCrop((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

weather_classes = [
    "dew",
    "fogsmog",
    "frost",
    "glaze",
    "hail",
    "lightning",
    "rain",
    "rainbow",
    "rime",
    "sandstrom",
    "snow",
]

weather_aliases = {
    "dew": "Berembun",
    "fogsmog": "Berkabut",
    "frost": "Embun Beku",
    "glaze": "Hujan Es",
    "hail": "Hujan Batu Es",
    "lightning": "Petir",
    "rain": "Hujan",
    "rainbow": "Pelangi",
    "rime": "Embun Es",
    "sandstrom": "Badai Pasir",
    "snow": "Salju",
}


weather_descriptions = {
    "Berembun": "Embun adalah fenomena meteorologi yang terjadi ketika uap air di udara mengembun pada permukaan yang dingin. Ini adalah proses alami yang terjadi di berbagai tempat di seluruh dunia, terutama di daerah yang memiliki kelembapan tinggi dan suhu dingin. Embun terbentuk ketika udara yang lembap bersentuhan dengan permukaan yang lebih dingin dari udara sekitarnya. Ketika udara lembap mendingin, kemampuannya untuk menahan uap air berkurang. Akibatnya, uap air tersebut mengembun menjadi tetesan air kecil yang menempel pada permukaan yang dingin.",
    
    "Berkabut": "Kabut asap adalah fenomena yang terjadi ketika asap dari kebakaran hutan atau lahan bercampur dengan kabut atau uap air di udara. Kondisi ini dapat menyebabkan penurunan jarak pandang yang signifikan, sehingga berbahaya bagi transportasi dan kesehatan manusia. Penyebab utamanya adalah kebakaran hutan dan lahan yang sering terjadi di musim kemarau, diperparah oleh cuaca kering, angin kencang, dan aktivitas manusia seperti pembakaran lahan. Dampaknya sangat luas, mulai dari masalah pernapasan dan penyakit jantung akibat partikel halus dalam asap, kerusakan ekosistem, hingga kerugian ekonomi akibat gangguan transportasi dan pariwisata.",
    
    "Embun Beku": "Embun beku, juga dikenal sebagai frost atau ibun, adalah fenomena alam yang terjadi ketika uap air di udara membeku pada permukaan padat yang suhunya di bawah titik beku. Fenomena ini sering terjadi di daerah beriklim sedang dan dingin, tetapi juga dapat terjadi di daerah tropis seperti Dataran Tinggi Dieng di Jawa Tengah. Embun beku terbentuk ketika udara lembap bersentuhan dengan permukaan yang suhunya di bawah titik beku. Uap air dalam udara akan mengembun menjadi tetesan air kecil pada permukaan tersebut. Ketika suhu permukaan terus menurun, tetesan air ini akan membeku dan membentuk kristal es yang tipis.",
    
    "Hujan Es": "Hujan es, atau dalam istilah ilmiahnya hail, adalah fenomena alam yang terjadi ketika tetesan air di dalam awan badai membeku dan membentuk bola-bola es yang kemudian jatuh ke bumi. Fenomena ini sering terjadi di daerah beriklim sedang dan dingin, tetapi juga dapat terjadi di daerah tropis seperti Indonesia. Hujan es adalah fenomena alam yang terjadi ketika tetesan air di dalam awan badai membeku dan membentuk bola-bola es yang kemudian jatuh ke bumi. Fenomena ini dapat memiliki dampak negatif yang signifikan, terutama bagi tanaman dan infrastruktur.",
    
    "Hujan Batu Es": "Hujan batu es, atau yang lebih dikenal dengan hujan es, adalah fenomena alam yang terjadi ketika tetesan air di dalam awan badai membeku dan membentuk bola-bola es yang kemudian jatuh ke bumi. Fenomena ini mirip dengan hujan es, tetapi dengan ukuran bola es yang lebih besar dan lebih padat. Hujan batu es biasanya terjadi di daerah beriklim sedang dan dingin, tetapi juga dapat terjadi di daerah tropis seperti Indonesia. Hujan batu es terbentuk di dalam awan cumulonimbus, yang merupakan awan badai yang menjulang tinggi ke atmosfer.",
    
    "Petir": "Petir adalah fenomena alam yang menakjubkan dan sekaligus menakutkan. Ini adalah pelepasan muatan listrik statis yang terjadi di atmosfer, biasanya selama badai petir. Petir terbentuk ketika awan badai, yang disebut awan cumulonimbus, mengumpulkan muatan listrik. Ketika tetesan air di dalam awan naik lebih tinggi, suhunya akan turun di bawah titik beku dan membentuk kristal es. Gesekan antara kristal es ini menyebabkan pemisahan muatan, dengan bagian atas awan memiliki muatan positif dan bagian bawah awan memiliki muatan negatif. Ketika perbedaan potensial antara muatan positif dan negatif cukup besar, terjadi pelepasan muatan listrik yang kita kenal sebagai petir.",
    
    "Hujan": "Hujan adalah fenomena alam yang terjadi ketika uap air di atmosfer terkondensasi menjadi tetesan air yang cukup berat untuk jatuh ke bumi. Proses ini dimulai dengan penguapan air dari permukaan bumi, seperti laut, sungai, dan danau, yang kemudian naik ke atmosfer bersama udara hangat. Di atmosfer, uap air tersebut mendingin dan mengembun menjadi tetesan air yang membentuk awan. Ketika tetesan air ini semakin besar dan berat, mereka akan jatuh ke bumi sebagai hujan. Ukuran tetesan air hujan bervariasi, mulai dari lonjong, berbentuk panekuk, hingga bola-bola kecil.",
    
    "Pelangi": "Pelangi adalah fenomena alam yang menakjubkan yang muncul sebagai busur warna-warni di langit, biasanya setelah hujan. Terbentuknya pelangi merupakan hasil interaksi antara sinar matahari, tetesan air hujan, dan udara. Ketika sinar matahari melewati tetesan air hujan, cahaya putih matahari dipecah menjadi berbagai warna karena pembiasan dan pemantulan cahaya. Sinar matahari yang masuk ke tetesan air dibiaskan, atau dibelokkan, dan kemudian dipantulkan di bagian belakang tetesan air. Ketika cahaya keluar dari tetesan air, ia kembali dibiaskan, menghasilkan spektrum warna yang kita lihat sebagai pelangi. Warna-warna pelangi selalu tersusun dalam urutan yang sama, yaitu merah, jingga, kuning, hijau, biru, nila, dan ungu.",
    
    "Embun Es": "Embun es, atau yang juga dikenal sebagai embun beku atau frost, adalah fenomena alam yang terjadi ketika uap air di udara membeku langsung menjadi kristal es kecil pada permukaan benda, seperti daun, rumput, atau tanah. Fenomena ini terjadi pada malam hari ketika suhu udara turun di bawah titik beku (0 derajat Celcius) dan kelembaban udara cukup tinggi. Embun es umumnya terbentuk di daerah beriklim dingin atau di daerah dataran tinggi, seperti di Dieng, Jawa Tengah. Proses terbentuknya embun es dimulai dengan uap air yang terkondensasi menjadi tetesan air kecil pada permukaan benda yang dingin. Ketika suhu udara semakin turun, tetesan air tersebut membeku dan membentuk kristal es. ",
    
    "Badai Pasir": "Badai pasir adalah fenomena alam yang terjadi ketika angin kencang mengangkat pasir dan debu dalam jumlah besar dari permukaan tanah yang kering. Badai pasir umumnya terjadi di wilayah gurun atau semi-gurun, di mana tanahnya kering dan sedikit vegetasi. Angin kencang yang mengangkat pasir dan debu dapat membentuk dinding pasir yang menjulang tinggi dan bergerak dengan cepat, menutupi area yang luas. Badai pasir dapat berlangsung selama beberapa menit hingga beberapa jam, dan dapat menyebabkan kerusakan pada properti, gangguan transportasi, dan bahaya kesehatan bagi manusia. Debu dan pasir yang terbawa angin dapat menyebabkan gangguan pernapasan, iritasi mata, dan masalah kesehatan lainnya.",
    
    "Salju": "Salju adalah fenomena alam yang terjadi ketika uap air di atmosfer membeku menjadi kristal es kecil yang kemudian jatuh ke bumi. Proses ini dimulai dengan uap air yang terkondensasi menjadi tetesan air kecil di dalam awan. Ketika suhu udara di dalam awan turun di bawah titik beku (0 derajat Celcius), tetesan air tersebut membeku dan membentuk kristal es. Kristal es ini kemudian akan bergabung dengan kristal es lainnya dan membentuk kepingan salju yang lebih besar. Kepingan salju ini kemudian akan jatuh ke bumi sebagai salju. Bentuk kepingan salju sangat beragam, tergantung pada suhu udara dan kelembaban udara saat pembentukannya. Salju umumnya terjadi di daerah beriklim dingin, seperti di negara-negara di belahan bumi utara dan selatan. Salju dapat menyebabkan berbagai dampak, seperti gangguan transportasi, kerusakan tanaman, dan masalah kesehatan bagi manusia."
}



weather_do = {
    "Berembun": "\n* Menikmati pemandangan pagi yang segar.\n* Mengambil foto keindahan alam.\n* Berjalan kaki di taman.\n* Mengadakan piknik ringan.\n* Menggunakan sepeda untuk berolahraga.",
    
    "Berkabut": "\n* Mengemudikan kendaraan dengan hati-hati dan kecepatan rendah.\n* Menikmati suasana tenang dan misterius.\n* Mengambil foto dengan efek kabut.\n* Menghadiri acara indoor.\n* Berjalan di area yang aman dan sudah dikenal.",
    
    "Embun Beku": "\n* Menikmati keindahan embun beku di pagi hari.\n* Mengambil foto embun yang membeku.\n* Menggunakan pakaian hangat saat keluar Mengadakan aktivitas indoor.\n* Menjaga tanaman dengan penutup yang sesuai.",
    
    "Hujan Es": "\n* Menyaksikan hujan es dari dalam rumah.\n* Menggunakan payung atau jas hujan jika keluar.\n* Menghindari perjalanan jauh saat cuaca buruk.\n* Memastikan kendaraan dalam kondisi baik.\n* Mengambil foto hujan es dari jarak aman.",
    
    "Hujan Batu Es": "\n* Mencari tempat berlindung saat hujan batu es.\n* Memastikan kendaraan dilindungi.\n* Menjaga jarak aman dari jendela saat hujan batu es.\n* Mengamati dari dalam rumah.\n* Melaporkan kerusakan jika terjadi.",
    
    "Petir": "\n* Mencari tempat berlindung yang aman.\n* Menyaksikan petir dari dalam rumah.\n* Menghindari penggunaan alat elektronik.\n* Menghindari berada di tempat terbuka.\n* Memastikan hewan peliharaan berada di dalam ruangan.",
    
    "Hujan": "\n* Menggunakan payung atau jas hujan.\n* Menikmati suara hujan dari dalam rumah.\n* Melakukan aktivitas indoor.\n* Mengambil foto hujan yang indah.\n* Menyiram tanaman jika diperlukan setelah hujan.",
    
    "Pelangi": "\n* Mengambil foto pelangi yang indah.\n* Menikmati cuaca setelah hujan.\n* Berjalan-jalan di luar saat cuaca cerah.\n* Mengadakan kegiatan outdoor.\n* Mempelajari fenomena alam.",
    
    "Embun Es": "\n* Mengambil foto embun es di pagi hari.\n* Menikmati suasana tenang di luar.\n* Berjalan di jalan yang aman.\n* Menggunakan pakaian hangat.\n* Melindungi tanaman dari embun es.",
    
    "Badai Pasir": "\n* Mencari tempat berlindung saat badai pasir.\n* Menjaga kesehatan dengan minum air yang cukup.\n* Memastikan kendaraan dalam keadaan baik.\n* Menggunakan masker jika terpaksa keluar.\n* Menghindari aktivitas luar ruangan.",
    
    "Salju": "\n* Bermain salju dengan aman.\n* Menggunakan pakaian hangat dan tahan air.\n* Mengemudikan kendaraan dengan hati-hati.\n* Mengambil foto pemandangan salju.\n* Mengadakan kegiatan ski atau snowboarding.",
}


weather_dont = {
    "Berembun": "\n* Mengemudikan kendaraan dengan kecepatan tinggi, terutama di jalan licin.\n* Mengabaikan kesehatan, seperti tidak menggunakan jaket jika merasa dingin.\n* Menggunakan alat elektronik di luar tanpa perlindungan.\n* Beraktivitas berat jika cuaca terlalu dingin Meninggalkan barang berharga di luar.",
    
    "Berkabut": "\n* Mengemudikan kendaraan tanpa lampu depan yang menyala.\n* Berjalan di tepi jalan yang sibuk.\n* Mengandalkan pemandangan jauh, karena visibilitas rendah.\n* Mengabaikan peringatan cuaca.\n* Melakukan aktivitas luar ruangan yang memerlukan penglihatan yang jelas.",
    
    "Embun Beku": "\n* Berjalan di permukaan yang mungkin licin.\n* Mengabaikan risiko hipotermia.\n* Meninggalkan hewan peliharaan di luar tanpa perlindungan.\n* Mengemudikan kendaraan tanpa memeriksa kondisi jalan.\n* Menggunakan peralatan listrik yang tidak tahan air di luar.",
    
    "Hujan Es": "\n* Mengemudikan kendaraan di jalan yang licin.\n* Berada di luar tanpa perlindungan saat hujan es.\n* Mengabaikan peringatan cuaca ekstrem.\n* Mencoba menangkap es yang jatuh.\n* Berjalan di bawah pohon atau struktur yang berisiko.",
    
    "Hujan Batu Es": "\n* Berada di luar saat hujan batu es.\n* Mengemudikan kendaraan tanpa perlindungan yang tepat.\n* Mengabaikan peringatan dari otoritas terkait.\n* Membiarkan anak-anak bermain di luar.\n* Menggunakan kendaraan yang tidak aman.",
    
    "Petir": "\n* Berdiri di bawah pohon atau struktur tinggi.\n* Menggunakan ponsel atau alat listrik saat ada petir.\n* Berenang atau berada di dekat kolam.\n* Mengemudikan kendaraan tanpa memeriksa cuaca.\n* Mengabaikan peringatan tentang badai petir.",
    
    "Hujan": "\n* Mengemudikan kendaraan dengan kecepatan tinggi.\n* Berjalan di jalan yang tergenang air.\n* Mengabaikan potensi banjir Menggunakan perangkat elektronik di luar.\n* Meninggalkan barang berharga di luar.",
    
    "Pelangi": "\n* Mengabaikan cuaca yang masih berpotensi hujan.\n* Berada di area yang berbahaya saat mengambil foto.\n* Mengemudikan kendaraan dengan sembrono.\n* Meninggalkan kendaraan tanpa pengawasan.\n* Mengabaikan peringatan cuaca.",
    
    "Embun Es": "\n* Berjalan di permukaan yang berisiko licin.\n* Mengabaikan risiko kesehatan akibat suhu rendah.\n* Meninggalkan hewan peliharaan di luar tanpa perlindungan.\n* Mengemudikan kendaraan tanpa memeriksa kondisi.\n* Menggunakan peralatan listrik yang tidak tahan air.",
    
    "Badai Pasir": "\n* Mengemudikan kendaraan tanpa perhatian.\n* Berada di luar tanpa perlindungan.\n* Mengabaikan peringatan badai pasir.\n* Mengizinkan anak-anak berada di luar.\n* Menggunakan alat elektronik yang tidak tahan debu.",
    
    "Salju": "\n* Mengemudikan kendaraan tanpa persiapan yang tepat.\n* Berada di luar tanpa perlindungan yang cukup.\n* Mengabaikan risiko hipotermia.\n* Melakukan aktivitas berbahaya di salju.\n* Meninggalkan hewan peliharaan di luar tanpa perlindungan.",
}



def predict_weather(image):
    try:
        model_path = 'weather_image_mobilenet_classifier.pt'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        print("Model berhasil dimuat.")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        image = preprocess(image).to(device)
        image = image.unsqueeze(0)

        with torch.no_grad():
            outputs = model(image)
            _, predicted_class = torch.max(outputs, 1)
        predicted_weather = weather_classes[predicted_class.item()]
        user_friendly_prediction = weather_aliases.get(predicted_weather, predicted_weather)
        return user_friendly_prediction

    except FileNotFoundError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


image_uploaded = False




logo_image = Image.open("logo.png")
st.sidebar.image(logo_image, use_container_width=True)

st.sidebar.subheader("Perhatian:")
st.sidebar.success("WeaVis adalah website yang dibangun bukan untuk tujuan komersil. Dilarang mengunggah dan menggunakan foto yang melanggar SARA dan berbau pornografi.  ")

st.sidebar.markdown("")

st.sidebar.subheader("Pilih Metode:")
method = st.sidebar.radio("Pilih bagaimana Anda ingin mengenali cuaca", ("Upload Gambar", "Gunakan Kamera"))

st.sidebar.markdown("")


if method == "Upload Gambar":
    st.sidebar.subheader("Unggah Gambar")
    uploaded_file = st.sidebar.file_uploader("Pilih gambar yang diinginkan.", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_uploaded = True
        st.image(image, use_container_width=True)
        st.markdown("""
            <div style="height: 20px;"></div>
        """, unsafe_allow_html=True) 

        with st.spinner('Sedang memproses...'):
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.05)
                my_bar.progress(percent_complete + 1)
            
            st.markdown("""
            <div style="height: 20px;"></div>
        """, unsafe_allow_html=True)     
            
            prediction = predict_weather(image)

        if isinstance(prediction, str) and prediction.startswith("Error:"):
            st.error(prediction)
        else:
            st.info(f"Cuaca Terdeteksi: **{prediction}**")
            st.markdown("")
            st.markdown(f"**Penjelasan:**")
            st.markdown(weather_descriptions.get(prediction, 'Tidak ada deskripsi untuk cuaca ini.'))
            st.markdown("")
            st.markdown(f"**Apa yang Boleh Dilakukan:** {weather_do.get(prediction, 'Ada kesalahan dalam menampilkan data.')}")
            st.markdown("")
            st.markdown(f"**Apa yang Tidak Boleh Dilakukan:** {weather_dont.get(prediction, 'Ada kesalahan dalam menampilkan data.')}")
            
    st.sidebar.markdown("")
    
    st.sidebar.subheader("Panduan:")
    st.sidebar.markdown("""
    - Pilih gambar yang diinginkan
    - Tunggu WeaVis memproses gambar
    - Lihat hasil cuaca terdeteksi
    - Baca informasi seputar cuaca                 
    
    """)


elif method == "Gunakan Kamera":
    image_uploaded = True
    st.markdown("""
        <div style="height: 20px;"></div>
    """, unsafe_allow_html=True) 
    
    st.sidebar.subheader("Panduan:")
    st.sidebar.markdown("""
    - Persiapkan kamera milikmu
    - Arahkan kamera dengan benar
    - Tekan tombol "Take Picture"
    - Tunggu WeaVis memproses gambar
    - Lihat hasil cuaca terdeteksi
    - Baca informasi seputar cuaca  
    
    
    """)
    
    enable = st.checkbox("Enable camera")
    picture = st.camera_input("Take a picture", disabled=not enable)

    if  picture:
        st.markdown("""
            <div style="height: 20px;"></div>
        """, unsafe_allow_html=True) 
        

        with st.spinner('Sedang memproses...'):
            # Simulasi proses yang memakan waktu
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.05)
                my_bar.progress(percent_complete + 1)
            
            st.markdown("""
            <div style="height: 20px;"></div>
        """, unsafe_allow_html=True)     
            prediction = predict_weather(Image.open(BytesIO(picture.getvalue())))

        if isinstance(prediction, str) and prediction.startswith("Error:"):
            st.error(prediction)
        else:
            st.info(f"Cuaca Terdeteksi: **{prediction}**")
            st.markdown("")
            st.markdown(f"**Penjelasan:**")
            st.markdown(weather_descriptions.get(prediction, 'Tidak ada deskripsi untuk cuaca ini.'))
            st.markdown("")
            st.markdown(f"**Apa yang Boleh Dilakukan:** {weather_do.get(prediction, 'Ada kesalahan dalam menampilkan data.')}")
            st.markdown("")
            st.markdown(f"**Apa yang Tidak Boleh Dilakukan:** {weather_dont.get(prediction, 'Ada kesalahan dalam menampilkan data.')}")
            
            
if not image_uploaded:
    st.title("Selamat Datang Di WeaVis, Bro!")
    st.subheader("Website Pengenalan Cuaca Berdasarkan Foto.")

    st.markdown("""
        <div style="height: 100px;"></div>
    """, unsafe_allow_html=True) 

    st.markdown("WeaVis adalah sebuah website pengenalan cuaca yang dapat membantu kamu mengenali cuaca yang sedang terjadi dan menampilkan informasi seputar cuaca tersebut. Informasi yang akan ditampilkan mencakup deskripsi cuaca, apa yang boleh dan tidak boleh dilakukan saat cuaca terjadi, penanganan terkait cuaca, dan berbagai informasi lainnya. WeaVis hadir dengan dua opsi pengenalan cuaca, yaitu melalui gambar yang diunggah, dan juga kamera yang secara langsung dapat mengambil foto cuaca terkini yang terjadi. WeaVis diharapkan dapat membantu penggunanya untuk memperoleh informasi seputar cuaca yang sedang terjadi. Selamat menggunakan!")

    st.markdown("""
        <div style="height: 100px;"></div>
    """, unsafe_allow_html=True) 

    st.markdown("Dibuat oleh [Kelompok 15](https://docs.google.com/spreadsheets/d/1kWOsguB6gWW5dBPbkkw1AN8n3dHMTLUHsfM8UPJ2dgY/edit?gid=0#gid=0) TPL A 59, Sekolah Vokasi IPB.")

