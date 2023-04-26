from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from func import *

UPLOAD_FOLDER = r'static/video/'
app = Flask(__name__, template_folder='templates')
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload')
def upload_file():
   return render_template('upload.html')

@app.route('/lab', methods = ['POST'])
def upload_files():
   if request.method == 'POST':
       video = request.files['video']
       filename = secure_filename(video.filename)
       video.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      
       video_path = r'static/video/'+str(filename)

       fr = Finger(video_path)

       red_dc1,red_dc,blue_dc,red_ac,blue_ac = fr.Raw_PPG()

       HR = fr.Find_HR_RR(red_dc,freqMin = 0.38, freqMax = 2,fps=30)
        
       RR = fr.Find_HR_RR(red_dc,freqMin = 0.1,freqMax = 0.3,fps=30)

       SPO2 = fr.Find_SpO2(red_dc,blue_dc,red_ac,blue_ac)

       HRV = fr.Find_HRV(red_dc,freqMin = 0.38, freqMax = 2,fps=30)

       SI = fr.Find_Stree(HRV)

       #SBP,DBP = fr.Find_BP(red_dc1,freqMin = 0.1, freqMax = 8,fps=125)
       #fr.close()
       #os.remove(video_path)

       return "HR : "+ str(HR) + " BPM"  "\n"  "RR : " + str(RR) + " RPM"  "\n"  "SPO2 : " + str(SPO2) + " %"  "\n"  "HRV : " + str(HRV) + " ms" "\n"  "Stress : " + SI    # + "\n" "SBP :"  + str(SBP) + "\nDBP : " + str(DBP)

if __name__ == "__main__":
    app.run(debug=True)
