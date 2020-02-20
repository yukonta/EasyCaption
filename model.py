from PIL import Image
import torch
import torchvision.transforms as transforms

import time
import re

from  image_captioning_my import CaptionNet, generate_caption

max_caption_len = 100
emb_size= 300
embed_usepretrained = False
lstm_num_layers=2
lstm_num = 2
lstm_dropout = 0.3
model_file_name = 'saved_models/' + 'model_img_capt.pt'
tokens_file_name = 'saved_models/' + 'tokens_img_capt.txt'
embed_weights_matrix_file_name = 'saved_models/' + 'embed_weights_matrix.npy'

    # В данном классе мы проводим обработку картинок, которые поступают к нам из телеграма.

class ImageCaptioningModel:
    def __init__(self):
        use_gpu = torch.cuda.is_available()
        if not use_gpu:
            self.device = 'cpu'
            print('CUDA is not available.  Use CPU ...')
        else:
            self.device = 'cuda'
            print('CUDA is available!  Use GPU ...')

        print(self.device)

        imsize = 299
        self.loader = transforms.Compose([
            transforms.Resize((imsize,imsize)),  # нормируем размер изображения
        ])  # превращаем в удобный формат

        pass

    def image_captioning(self, img_stream):
        #param: img_stream: поток байтов входного изображения
        #output: сгенерированные текстовые описания изображения
        print('image captioning ', model_file_name)

        out_captions = self.process_image(img_stream)

        return out_captions

    def process_image(self, img_stream):
        # param: img_stream: поток байтов входного изображения
        # output: out_captions: сгенерированные текстовые описания изображения

        print('process_image ')

        image = Image.open(img_stream)
        image = self.loader(image)

        out_captions = self.make_captions(content_img=image)

        return out_captions

    def make_captions(self, content_img):
        # param: content_img: входное изображение
        # output: out_captions: сгенерированные текстовые описания изображения

        print('1')
        now = time.time()
        fn = 'im_file ' + str(now) + '.jpg'
        print(fn)
        content_img.save(fn)
        print('2')
        content_image = content_img

        content_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        content_image = content_transform(content_image)
        print('3')
        f_tokens = open(tokens_file_name, 'r')
        tokens = f_tokens.readline()
        f_tokens.close()
        tokens = tokens.split(' ')

        capt_model = CaptionNet(n_tokens=len(tokens), emb_size=emb_size, lstm_num_layers=lstm_num_layers, lstm_num=lstm_num,
                     lstm_dropout=lstm_dropout,
                     embed_usepretrained=embed_usepretrained,
                     embed_weights_matrix = None)

        state_dict = torch.load(model_file_name)
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        capt_model.load_state_dict(state_dict)
        capt_model.to(self.device)
        print('4')

        out_captions = ''
        for i in range(3):
            caption = generate_caption(capt_model, content_image, tokens, t=5., max_len=max_caption_len, device=self.device)[0]
            caption = ' '.join(caption.split(' ')[1:-1])
            out_captions = out_captions + '*** ' + caption
        print(out_captions)
        print('5')
        return out_captions
        # utils.save_image(args.output_image, output[0])


