import sys
import getopt

import torch
from tqdm import tqdm

import utils
import models

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

def predict_model(model, predict_loader):
    model.eval()
    ret = []
    with torch.no_grad():
        with tqdm(predict_loader) as t:
            for data, _ in t:
                data = data.to(DEVICE)
                output = model(data).data
                for each in output:
                    ret.append(float(each))
    return ret

def predict(predict_csv_path, output_csv_path, model_path, batch_size, input_length, window_size):
    predict_loader, file_path_list = utils.get_data_loader(predict_csv_path, batch_size, False, True)
    model = models.PNN(input_length, window_size).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    predictions = predict_model(model, predict_loader)
    with open(output_csv_path, 'w') as f:
        for fn, prediction in zip(file_path_list, predictions):
            f.write('{},{}\n'.format(fn, prediction))

def main(argv):
    try:
        predict_csv_path = None
        output_csv_path = None
        model_path = None
        batch_size = 128
        input_length = 1600
        window_size = 5
        optlist, args = getopt.getopt(argv[1:], '', ['help', 'predict=', 'model=', 'batch_size=', 'output=', 'input_length=', 'window_size='])
        for opt, arg in optlist:
            if opt == '--help':
                utils.help()
                sys.exit(0)
            elif opt == '--predict':
                predict_csv_path = arg
            elif opt == '--model':
                model_path = arg
            elif opt == '--batch_size':
                batch_size = int(arg)
            elif opt == '--output':
                output_csv_path = arg
            elif opt == '--input_length':
                input_length = int(arg)
            elif opt == '--window_size':
                window_size = int(arg)
        if predict_csv_path == None:
            print('The following values must be input')
            print('predict')
            utils.help()
            sys.exit(1)
        predict(predict_csv_path, output_csv_path, model_path, batch_size, input_length, window_size)
    except Exception as e:
        print(e)
        sys.exit(1)

if __name__ == '__main__':
    main(sys.argv)