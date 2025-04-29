import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import requests
import os
import time
from io import BytesIO


torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

preprocess = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.CenterCrop((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

weather_classes = [
    "dew",
    "fogsmog",
    "lightning",
    "rain",
    "rainbow",
    "sandstrom",
    "snow",
    "sunny"
]

weather_aliases = {
    "dew": "Berembun",
    "fogsmog": "Berkabut",
    "lightning": "Petir",
    "rain": "Hujan",
    "rainbow": "Pelangi",
    "sandstrom": "Badai Pasir",
    "snow": "Salju",
    "sunny": "Cerah"
}


weather_descriptions = {
    "Berembun": "Embun adalah fenomena meteorologi yang terjadi ketika uap air di udara mengembun pada permukaan yang dingin. Ini adalah proses alami yang terjadi di berbagai tempat di seluruh dunia, terutama di daerah yang memiliki kelembapan tinggi dan suhu dingin. Embun terbentuk ketika udara yang lembap bersentuhan dengan permukaan yang lebih dingin dari udara sekitarnya. Ketika udara lembap mendingin, kemampuannya untuk menahan uap air berkurang. Akibatnya, uap air tersebut mengembun menjadi tetesan air kecil yang menempel pada permukaan yang dingin.",
    
    "Berkabut": "Kabut asap adalah fenomena yang terjadi ketika asap dari kebakaran hutan atau lahan bercampur dengan kabut atau uap air di udara. Kondisi ini dapat menyebabkan penurunan jarak pandang yang signifikan, sehingga berbahaya bagi transportasi dan kesehatan manusia. Penyebab utamanya adalah kebakaran hutan dan lahan yang sering terjadi di musim kemarau, diperparah oleh cuaca kering, angin kencang, dan aktivitas manusia seperti pembakaran lahan. Dampaknya sangat luas, mulai dari masalah pernapasan dan penyakit jantung akibat partikel halus dalam asap, kerusakan ekosistem, hingga kerugian ekonomi akibat gangguan transportasi dan pariwisata.",
    
    "Petir": "Petir adalah fenomena alam yang menakjubkan dan sekaligus menakutkan. Ini adalah pelepasan muatan listrik statis yang terjadi di atmosfer, biasanya selama badai petir. Petir terbentuk ketika awan badai, yang disebut awan cumulonimbus, mengumpulkan muatan listrik. Ketika tetesan air di dalam awan naik lebih tinggi, suhunya akan turun di bawah titik beku dan membentuk kristal es. Gesekan antara kristal es ini menyebabkan pemisahan muatan, dengan bagian atas awan memiliki muatan positif dan bagian bawah awan memiliki muatan negatif. Ketika perbedaan potensial antara muatan positif dan negatif cukup besar, terjadi pelepasan muatan listrik yang kita kenal sebagai petir.",
    
    "Hujan": "Hujan adalah fenomena alam yang terjadi ketika uap air di atmosfer terkondensasi menjadi tetesan air yang cukup berat untuk jatuh ke bumi. Proses ini dimulai dengan penguapan air dari permukaan bumi, seperti laut, sungai, dan danau, yang kemudian naik ke atmosfer bersama udara hangat. Di atmosfer, uap air tersebut mendingin dan mengembun menjadi tetesan air yang membentuk awan. Ketika tetesan air ini semakin besar dan berat, mereka akan jatuh ke bumi sebagai hujan. Ukuran tetesan air hujan bervariasi, mulai dari lonjong, berbentuk panekuk, hingga bola-bola kecil.",
    
    "Pelangi": "Pelangi adalah fenomena alam yang menakjubkan yang muncul sebagai busur warna-warni di langit, biasanya setelah hujan. Terbentuknya pelangi merupakan hasil interaksi antara sinar matahari, tetesan air hujan, dan udara. Ketika sinar matahari melewati tetesan air hujan, cahaya putih matahari dipecah menjadi berbagai warna karena pembiasan dan pemantulan cahaya. Sinar matahari yang masuk ke tetesan air dibiaskan, atau dibelokkan, dan kemudian dipantulkan di bagian belakang tetesan air. Ketika cahaya keluar dari tetesan air, ia kembali dibiaskan, menghasilkan spektrum warna yang kita lihat sebagai pelangi. Warna-warna pelangi selalu tersusun dalam urutan yang sama, yaitu merah, jingga, kuning, hijau, biru, nila, dan ungu.",
    
    "Badai Pasir": "Badai pasir adalah fenomena alam yang terjadi ketika angin kencang mengangkat pasir dan debu dalam jumlah besar dari permukaan tanah yang kering. Badai pasir umumnya terjadi di wilayah gurun atau semi-gurun, di mana tanahnya kering dan sedikit vegetasi. Angin kencang yang mengangkat pasir dan debu dapat membentuk dinding pasir yang menjulang tinggi dan bergerak dengan cepat, menutupi area yang luas. Badai pasir dapat berlangsung selama beberapa menit hingga beberapa jam, dan dapat menyebabkan kerusakan pada properti, gangguan transportasi, dan bahaya kesehatan bagi manusia. Debu dan pasir yang terbawa angin dapat menyebabkan gangguan pernapasan, iritasi mata, dan masalah kesehatan lainnya.",
    
    "Salju": "Salju adalah fenomena alam yang terjadi ketika uap air di atmosfer membeku menjadi kristal es kecil yang kemudian jatuh ke bumi. Proses ini dimulai dengan uap air yang terkondensasi menjadi tetesan air kecil di dalam awan. Ketika suhu udara di dalam awan turun di bawah titik beku (0 derajat Celcius), tetesan air tersebut membeku dan membentuk kristal es. Kristal es ini kemudian akan bergabung dengan kristal es lainnya dan membentuk kepingan salju yang lebih besar. Kepingan salju ini kemudian akan jatuh ke bumi sebagai salju. Bentuk kepingan salju sangat beragam, tergantung pada suhu udara dan kelembaban udara saat pembentukannya. Salju umumnya terjadi di daerah beriklim dingin, seperti di negara-negara di belahan bumi utara dan selatan. Salju dapat menyebabkan berbagai dampak, seperti gangguan transportasi, kerusakan tanaman, dan masalah kesehatan bagi manusia.",
    
    "Cerah": "Cuaca cerah adalah kondisi cuaca yang menyenangkan dan menenangkan. Langit tampak biru cerah tanpa awan atau hanya sedikit awan tipis yang melayang-layang. Matahari bersinar terang, memancarkan sinar hangat yang membuat suasana terasa nyaman. Udara terasa segar dan bersih, bebas dari polusi atau asap. Pada cuaca cerah, tidak ada tanda-tanda hujan, angin kencang, atau badai. Suhu udara biasanya hangat, cocok untuk berbagai aktivitas di luar ruangan seperti piknik, bermain di taman, atau sekadar bersantai di bawah sinar matahari. Cuaca cerah memberikan perasaan positif dan optimis. Banyak orang merasa senang dan bersemangat ketika cuaca cerah, karena dapat meningkatkan suasana hati dan memberikan energi positif.",
}



weather_do = {
    "Berembun": "\n* Menikmati pemandangan pagi yang segar.\n* Mengambil foto keindahan alam.\n* Berjalan kaki di taman.\n* Mengadakan piknik ringan.\n* Menggunakan sepeda untuk berolahraga.",
    
    "Berkabut": "\n* Mengemudikan kendaraan dengan hati-hati dan kecepatan rendah.\n* Menikmati suasana tenang dan misterius.\n* Mengambil foto dengan efek kabut.\n* Menghadiri acara indoor.\n* Berjalan di area yang aman dan sudah dikenal.",
    
    "Petir": "\n* Mencari tempat berlindung yang aman.\n* Menyaksikan petir dari dalam rumah.\n* Menghindari penggunaan alat elektronik.\n* Menghindari berada di tempat terbuka.\n* Memastikan hewan peliharaan berada di dalam ruangan.",
    
    "Hujan": "\n* Menggunakan payung atau jas hujan.\n* Menikmati suara hujan dari dalam rumah.\n* Melakukan aktivitas indoor.\n* Mengambil foto hujan yang indah.\n* Menyiram tanaman jika diperlukan setelah hujan.",
    
    "Pelangi": "\n* Mengambil foto pelangi yang indah.\n* Menikmati cuaca setelah hujan.\n* Berjalan-jalan di luar saat cuaca cerah.\n* Mengadakan kegiatan outdoor.\n* Mempelajari fenomena alam.",
    
    "Badai Pasir": "\n* Mencari tempat berlindung saat badai pasir.\n* Menjaga kesehatan dengan minum air yang cukup.\n* Memastikan kendaraan dalam keadaan baik.\n* Menggunakan masker jika terpaksa keluar.\n* Menghindari aktivitas luar ruangan.",
    
    "Salju": "\n* Bermain salju dengan aman.\n* Menggunakan pakaian hangat dan tahan air.\n* Mengemudikan kendaraan dengan hati-hati.\n* Mengambil foto pemandangan salju.\n* Mengadakan kegiatan ski atau snowboarding.",
    
    "Cerah": "\n* Nikmati sinar matahari.\n* Beraktivitas di luar ruangan.\n* Mengabadikan cuaca dengan berfoto.\n* Berkebun dan menanam tanaman dibawah matahari.\n* Piknik di taman bersama teman.",
}


weather_dont = {
    "Berembun": "\n* Mengemudikan kendaraan dengan kecepatan tinggi, terutama di jalan licin.\n* Mengabaikan kesehatan, seperti tidak menggunakan jaket jika merasa dingin.\n* Menggunakan alat elektronik di luar tanpa perlindungan.\n* Beraktivitas berat jika cuaca terlalu dingin Meninggalkan barang berharga di luar.",
    
    "Berkabut": "\n* Mengemudikan kendaraan tanpa lampu depan yang menyala.\n* Berjalan di tepi jalan yang sibuk.\n* Mengandalkan pemandangan jauh, karena visibilitas rendah.\n* Mengabaikan peringatan cuaca.\n* Melakukan aktivitas luar ruangan yang memerlukan penglihatan yang jelas.",
    
    "Petir": "\n* Berdiri di bawah pohon atau struktur tinggi.\n* Menggunakan ponsel atau alat listrik saat ada petir.\n* Berenang atau berada di dekat kolam.\n* Mengemudikan kendaraan tanpa memeriksa cuaca.\n* Mengabaikan peringatan tentang badai petir.",
    
    "Hujan": "\n* Mengemudikan kendaraan dengan kecepatan tinggi.\n* Berjalan di jalan yang tergenang air.\n* Mengabaikan potensi banjir.\n* Menggunakan perangkat elektronik di luar.\n* Meninggalkan barang berharga di luar.",
    
    "Pelangi": "\n* Mengabaikan cuaca yang masih berpotensi hujan.\n* Berada di area yang berbahaya saat mengambil foto.\n* Mengemudikan kendaraan dengan sembrono.\n* Meninggalkan kendaraan tanpa pengawasan.\n* Mengabaikan peringatan cuaca.",
    
    "Badai Pasir": "\n* Mengemudikan kendaraan tanpa perhatian.\n* Berada di luar tanpa perlindungan.\n* Mengabaikan peringatan badai pasir.\n* Mengizinkan anak-anak berada di luar.\n* Menggunakan alat elektronik yang tidak tahan debu.",
    
    "Salju": "\n* Mengemudikan kendaraan tanpa persiapan yang tepat.\n* Berada di luar tanpa perlindungan yang cukup.\n* Mengabaikan risiko hipotermia.\n* Melakukan aktivitas berbahaya di salju.\n* Meninggalkan hewan peliharaan di luar tanpa perlindungan.",
    
    "Cerah": "\n* Berjemur di bawah terik matahari selama waktu terpanas.\n* Lupa minum air putih dan dehidrasi.\n* Menggunakan pakaian gelap.\n* Mengabaikan kesehatan.\n* Meninggalkan barang berharga di luar ruangan.",
}


weather_funfact = {
    "Berembun": "Embun pagi yang sering kita lihat menempel di dedaunan bukanlah sekadar tetesan air biasa. Bayangkan jutaan tetesan air mungil, masing-masing seperti sebuah permata kecil yang berkilauan di bawah sinar matahari pagi. Proses terbentuknya embun itu sendiri cukup ajaib; uap air di udara mendingin dan mengembun di permukaan yang lebih dingin, seperti daun atau rumput, membentuk tetesan-tetesan kecil yang kemudian bergabung menjadi embun yang lebih besar. Lebih menarik lagi, embun ini berperan penting dalam kehidupan tumbuhan, menyediakan sumber air tambahan di pagi hari, layaknya sebuah minuman segar untuk tanaman! Jadi, lain kali kamu melihat embun pagi, bayangkanlah jutaan tetesan air mungil itu sebagai sebuah keajaiban alam mini.",
    
    "Berkabut": "Kabut, yang seringkali digambarkan sebagai pemandangan yang suram, menyimpan rahasia yang cukup menarik. Pernahkah kamu memperhatikan bagaimana suara-suara di sekitarmu terdengar lebih jelas dan nyaring saat berkabut? Ini karena kabut, yang terdiri dari tetesan air kecil yang melayang di udara, mampu menyerap dan membiaskan gelombang suara dengan cara yang unik, sehingga suara-suara tersebut seolah-olah dipantulkan kembali ke pendengaran kita. Selain itu, kabut juga menciptakan ilusi optik yang menarik. Objek yang berada di kejauhan terlihat lebih dekat dan kabur, menciptakan suasana misterius yang seringkali menginspirasi seniman dan penulis.",
    
    "Petir": "Kita semua tahu petir sebagai kilatan cahaya yang menakjubkan dan suara gemuruh yang menggelegar. Namun, tahukah kamu bahwa setiap sambaran petir sebenarnya menghasilkan gelombang radio yang kuat? Gelombang ini, meskipun tak terlihat, bisa ditangkap oleh radio dan menghasilkan suara kretek yang khas. Lebih unik lagi, energi listrik yang dihasilkan oleh satu sambaran petir cukup untuk menyalakan sebuah lampu 100 watt selama berbulan-bulan! Bayangkan kekuatan alam yang luar biasa terpendam dalam fenomena alam yang spektakuler ini.",
    
    "Hujan": "Hujan, sumber kehidupan bagi planet kita, tidak selalu jatuh dalam bentuk tetesan air yang biasa kita kenal. Di beberapa daerah dengan suhu yang sangat rendah, hujan bisa berbentuk butiran es kecil yang disebut hujan es, atau bahkan berupa salju yang leleh sebelum mencapai tanah. Bentuk dan ukuran tetesan hujan pun beragam, dipengaruhi oleh kecepatan angin dan ketinggian awan. Jadi, hujan bukanlah sekadar air yang jatuh dari langit, tetapi sebuah fenomena alam yang kompleks dan beragam.",
    
    "Pelangi": "Pelangi, simbol keindahan dan harapan, lebih dari sekadar busur warna-warni di langit. Tahukah kamu bahwa untuk melihat pelangi, kamu harus berada di antara matahari dan hujan? Matahari harus berada di belakangmu, dan hujan di depanmu. Dan, meskipun terlihat seperti busur, pelangi sebenarnya merupakan lingkaran penuh yang bagian bawahnya terhalang oleh cakrawala. Jadi, lain kali kamu melihat pelangi, ingatlah bahwa kamu sedang menyaksikan sebuah fenomena optik yang menakjubkan dan penuh misteri.",
    
    "Badai Pasir": "Badai pasir, yang seringkali digambarkan sebagai pemandangan yang dramatis dan menakutkan, memiliki kekuatan yang luar biasa. Angin kencang mampu mengangkat pasir dan debu hingga ketinggian yang sangat tinggi, menciptakan pemandangan yang spektakuler namun juga berbahaya. Lebih menakjubkan lagi, badai pasir mampu mengangkut pasir dan debu hingga ke tempat yang sangat jauh. Ada kasus di mana pasir dari gurun Sahara terbawa angin hingga ke Amerika Selatan!",
    
    "Salju": "Setiap kepingan salju unik, tidak ada dua kepingan salju yang sama persis. Bentuknya yang rumit dan indah terbentuk dari proses kristalisasi es di atmosfer, dipengaruhi oleh suhu, tekanan udara, dan kelembapan. Meskipun terlihat sederhana, salju menyimpan misteri yang kompleks dan menakjubkan.",
    
    "Cerah": "Cuaca cerah yang kita nikmati setiap hari menyimpan manfaat yang lebih dari sekadar suasana hati yang baik. Cahaya matahari yang melimpah membantu tubuh memproduksi vitamin D, yang penting untuk kesehatan tulang dan sistem kekebalan tubuh. Selain itu, cuaca cerah juga dapat meningkatkan produktivitas dan kreativitas. Jadi, nikmatilah cuaca cerah sebagai sebuah hadiah alam yang bermanfaat bagi kesehatan dan kesejahteraan kita.",
}




def predict_weather(image):
    try:
        model_path = 'model/weavis_mobilenetv3_large_model.pt'
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




logo_image = Image.open("asset/logo.png")
st.sidebar.image(logo_image, use_container_width=True)

st.sidebar.subheader("Perhatian:")
st.sidebar.warning("WeaVis adalah website yang dibangun bukan untuk tujuan komersil. Dilarang mengunggah dan menggunakan foto yang melanggar SARA dan berbau pornografi.  ")

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
            st.info(f"🔍 Cuaca Terdeteksi: **{prediction}**")
            st.divider()
            st.markdown(f"**Penjelasan:**")
            st.markdown(weather_descriptions.get(prediction, 'Tidak ada deskripsi untuk cuaca ini.'))
            st.markdown("")
            st.markdown(f"🏊‍♀️ **Apa yang Boleh Dilakukan:** {weather_do.get(prediction, 'Ada kesalahan dalam menampilkan data.')}")
            st.markdown("")
            st.markdown(f"⛔️ **Apa yang Tidak Boleh Dilakukan:** {weather_dont.get(prediction, 'Ada kesalahan dalam menampilkan data.')}")
            st.markdown("")
            st.markdown(f"**Fun Facts Seputar Cuaca:**")
            st.markdown(weather_funfact.get(prediction, 'Tidak ada deskripsi untuk cuaca ini.'))
            st.divider()
            st.caption(f"Sumber Informasi: https://www.cici.com")
            
            
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
            st.info(f"🔍 Cuaca Terdeteksi: **{prediction}**")
            st.divider()
            st.markdown(f"**Penjelasan:**")
            st.markdown(weather_descriptions.get(prediction, 'Tidak ada deskripsi untuk cuaca ini.'))
            st.markdown("")
            st.markdown(f"🏊‍♀️ **Apa yang Boleh Dilakukan:** {weather_do.get(prediction, 'Ada kesalahan dalam menampilkan data.')}")
            st.markdown("")
            st.markdown(f"⛔️ **Apa yang Tidak Boleh Dilakukan:** {weather_dont.get(prediction, 'Ada kesalahan dalam menampilkan data.')}")
            st.markdown("")
            st.markdown(f"**Fun Facts Seputar Cuaca:**")
            st.markdown(weather_funfact.get(prediction, 'Tidak ada deskripsi untuk cuaca ini.'))
            st.divider()
            st.caption(f"Sumber Informasi: https://www.cici.com")
            
            
if not image_uploaded:
    st.title("Selamat Datang Di WeaVis, Bro! 👋")
    st.subheader("Pelajari cuaca yang terjadi di sekitarmu.")
    
    st.divider()

    st.markdown("""
        <div style="height: 55px;"></div>
    """, unsafe_allow_html=True) 

    st.markdown("WeaVis adalah sebuah website pengenalan cuaca yang dapat membantu kamu mengenali cuaca yang sedang terjadi dan menampilkan informasi seputar cuaca tersebut. Informasi yang akan ditampilkan mencakup deskripsi cuaca, apa yang boleh dan tidak boleh dilakukan saat cuaca terjadi, penanganan terkait cuaca, dan berbagai informasi lainnya. WeaVis hadir dengan dua opsi pengenalan cuaca, yaitu melalui gambar yang diunggah, dan juga kamera yang secara langsung dapat mengambil foto cuaca terkini yang terjadi. WeaVis diharapkan dapat membantu penggunanya untuk memperoleh informasi seputar cuaca yang sedang terjadi. Selamat menggunakan!")

    st.markdown("""
        <div style="height: 100px;"></div>
    """, unsafe_allow_html=True) 

    st.markdown("Dibuat oleh [Kelompok 15](https://docs.google.com/spreadsheets/d/1kWOsguB6gWW5dBPbkkw1AN8n3dHMTLUHsfM8UPJ2dgY/edit?gid=0#gid=0) TPL A 59, Sekolah Vokasi IPB.")
