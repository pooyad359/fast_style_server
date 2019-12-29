import cv2
import torch
from torchvision import transforms
from transformer_net import TransformerNet
from imutils.video import VideoStream
from imutils import resize
import numpy as np
from flask import Flask, redirect, flash, request, url_for, send_file, jsonify
import os
from style_cam import style_frame, read_state_dict
import json


ALLOWED_EXTENSIONS = set(['bmp', 'png', 'jpg', 'jpeg', 'gif'])
app=Flask(__name__)
app.secret_key = "secret key"
path=os.path.abspath(os.curdir)

# Load the model
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('<<< Using CUDA >>>')
else:
    device = torch.device('cpu')
    print('CUDA device not available.')
    print('<<< Using CPU >>>')

weights = []
model_list={}
for idx,file in enumerate(os.listdir('models')):
    wts=read_state_dict(os.path.join(path,'models',file))
    weights.append(wts)
    model_list.update({file:idx})
model = TransformerNet()
model.to(device)


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST'])
def style():
    print([i for i in request.form.keys()])
    response = {'Status': None,
                'Message':None,
                'file':None}
    print(request.url)
    if 'file' not in request.files:
        response['Status'] = 'Failed'
        response['Message'] = 'No file attached to the request.'
        return json.dumps(response)
    file=request.files['file']
    if file.filename == '':
        response['Status'] = 'Failed'
        response['Message'] = 'No file attached to the request.'
        return json.dumps(response)
    if file and allowed_file(file.filename):
        filename = file.filename
        img_path=os.path.join(path,'static',filename)
        file.save(img_path)
        print(img_path)
        img=cv2.imread(img_path)
        print([k for k in request.form.keys()])
        style = request.form['style']
        print('\n\n',style,'\n\n')
        if style+'.pth' not in model_list.keys():
            idx_model = 0
        else:
            idx_model = model_list[style+'.pth']
        model.load_state_dict(weights[idx_model])
        output = style_frame(img,model,device)
        cv2.imwrite('result.jpg',output[:,:,::-1])
        return send_file('result.jpg')
    else:
        response['Status'] = 'Failed'
        response['Message'] = 'Unacceptable file type.'
        return json.dumps(response)

@app.route('/list',methods=['GET'])
def list_of_models():
    mlist = [os.path.splitext(m)[0] for m in model_list]
    return jsonify({'list':mlist})

if __name__=='__main__':
    print('Server starting')
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port,debug=True)

