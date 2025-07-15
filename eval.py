import os
import argparse
from tqdm import tqdm
from data.data import *
from torchvision import transforms
from torch.utils.data import DataLoader
from loss.losses import *
from net.CIDNet import CIDNet
eval_parser = argparse.ArgumentParser(description='Eval')
eval_parser.add_argument('--lol', action='store_true', help='output lolv1 dataset')
eval_parser.add_argument('--lol_v2_real', action='store_true', help='output lol_v2_real dataset')
eval_parser.add_argument('--lol_v2_syn', action='store_true', help='output lol_v2_syn dataset')
eval_parser.add_argument('--SICE_grad', action='store_true', help='output SICE_grad dataset')
eval_parser.add_argument('--SICE_mix', action='store_true', help='output SICE_mix dataset')

eval_parser.add_argument('--unpaired', action='store_true', help='output unpaired dataset')
eval_parser.add_argument('--DICM', action='store_true', help='output DICM dataset')
eval_parser.add_argument('--LIME', action='store_true', help='output LIME dataset')
eval_parser.add_argument('--MEF', action='store_true', help='output MEF dataset')
eval_parser.add_argument('--NPE', action='store_true', help='output NPE dataset')
eval_parser.add_argument('--VV', action='store_true', help='output VV dataset')
eval_parser.add_argument('--alpha', type=float, default=1.0)
eval_parser.add_argument('--gamma', type=float, default=1.0)

ep = eval_parser.parse_args()

def eval(model, testing_data_loader, model_path, output_folder,norm_size=True,LOL=False,v2=False,unpaired=False,alpha=1.0,gamma=1.0):
    torch.set_grad_enabled(False)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    print('Pre-trained model is loaded.')
    model.eval()
    print('Evaluation:')
    if LOL:
        model.trans.gated = True
    elif v2:
        model.trans.gated2 = True
        model.trans.alpha = alpha
    elif unpaired:
        model.trans.gated2 = True
        model.trans.alpha = alpha
    for batch in tqdm(testing_data_loader):
        with torch.no_grad():
            if norm_size:
                input, name = batch[0], batch[1]
            else:
                input, name, h, w = batch[0], batch[1], batch[2], batch[3]
            input = input.cuda()
            output = model(input**gamma) 
        if not os.path.exists(output_folder):          
            os.makedirs(output_folder, exist_ok=True) 
        output = torch.clamp(output.cuda(),0,1).cuda()
        if not norm_size:
            output = output[:, :, :h, :w]
        output_img = transforms.ToPILImage()(output.squeeze(0))
        output_img.save(output_folder + name[0])
        torch.cuda.empty_cache()
    print('===> End evaluation')
    if LOL:
        model.trans.gated = False
    elif v2:
        model.trans.gated2 = False
    torch.set_grad_enabled(True)
    
if __name__ == '__main__':
    gpu = "0"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, or need to change CUDA_VISIBLE_DEVICES number")
    
    if not os.path.exists('./output'):          
            os.mkdir('./output')  
    
    norm_size = True
    num_workers = 1
    alpha = None
    output_folder = ""
    weight_path = ""
    if ep.lol:
        eval_data = DataLoader(dataset=get_eval_set("../datasets/LOL/LOLv1/eval15/low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv1/'
        weight_path = '../weight/lolv1.pth'
    elif ep.lol_v2_real:
        eval_data = DataLoader(dataset=get_eval_set("../datasets/LOL/LOLv2/Real_captured/Test/Low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv2_real/'
        weight_path = '../weight/lolv2real.pth'
        alpha = 0.8
    elif ep.lol_v2_syn:
        eval_data = DataLoader(dataset=get_eval_set("../datasets/LOL/LOLv2/Synthetic/Test/Low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv2_syn/'
        weight_path = "../weight/lolv2syn.pth"
    elif ep.SICE_grad:
        eval_data = DataLoader(dataset=get_SICE_eval_set("../datasets/SICE/SICE_Grad"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/SICE_grad/'
        weight_path = '../weight/sice.pth'
        norm_size = False    
    elif ep.SICE_mix:
        eval_data = DataLoader(dataset=get_SICE_eval_set("../datasets/SICE/SICE_Mix"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/SICE_mix/'
        weight_path = '../weight/sice.pth'
        norm_size = False
    elif ep.unpaired: 
        if ep.DICM:
            eval_data = DataLoader(dataset=get_SICE_eval_set("../datasets/LOL/DICM"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/DICM/'
        elif ep.LIME:
            eval_data = DataLoader(dataset=get_SICE_eval_set("../datasets/LOL/LIME"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/LIME/'
        elif ep.MEF:
            eval_data = DataLoader(dataset=get_SICE_eval_set("../datasets/LOL/MEF"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/MEF/'
        elif ep.NPE:
            eval_data = DataLoader(dataset=get_SICE_eval_set("../datasets/LOL/NPE"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/NPE/'
        elif ep.VV:
            eval_data = DataLoader(dataset=get_SICE_eval_set("../datasets/LOL/VV"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = './output/VV/'
        alpha = ep.alpha
        norm_size = False
        weight_path = "../weight/unpair.pth"
        
    eval_net = CIDNet().cuda()
    eval(eval_net, eval_data, weight_path, output_folder,norm_size=norm_size,LOL=ep.lol,v2=ep.lol_v2_real,unpaired=ep.unpaired,alpha=alpha,gamma=ep.gamma)