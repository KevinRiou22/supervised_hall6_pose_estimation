# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import copy
import torch
import itertools
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from common.mocap_dataset import MocapDataset
from common.camera import *
from common.utils import wrap
from common.quaternion import qrot, qinverse
from common.loss import *
from common.multiview import Camera, project_3d_points_to_image_plane_without_distortion
from common.skeleton import Skeleton
from common.loss import mpjpe

h36m_skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                  16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                         joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                         joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])

h36m_cameras_intrinsic_params = [
    {
        'id': '54138969',
        'center': [512.54150390625, 515.4514770507812],
        'focal_length': [1145.0494384765625, 1143.7811279296875],
        'radial_distortion': [-0.20709891617298126, 0.24777518212795258, -0.0030751503072679043],
        'tangential_distortion': [-0.0009756988729350269, -0.00142447161488235],
        'res_w': 1000,
        'res_h': 1002,
        'azimuth': 70, # Only used for visualization
    },
    {
        'id': '55011271',
        'center': [508.8486328125, 508.0649108886719],
        'focal_length': [1149.6756591796875, 1147.5916748046875],
        'radial_distortion': [-0.1942136287689209, 0.2404085397720337, 0.006819975562393665],
        'tangential_distortion': [-0.0016190266469493508, -0.0027408944442868233],
        'res_w': 1000,
        'res_h': 1000,
        'azimuth': -70, # Only used for visualization
    },
    {
        'id': '58860488',
        'center': [519.8158569335938, 501.40264892578125],
        'focal_length': [1149.1407470703125, 1148.7989501953125],
        'radial_distortion': [-0.2083381861448288, 0.25548800826072693, -0.0024604974314570427],
        'tangential_distortion': [0.0014843869721516967, -0.0007599993259645998],
        'res_w': 1000,
        'res_h': 1000,
        'azimuth': 110, # Only used for visualization
    },
    {
        'id': '60457274',
        'center': [514.9682006835938, 501.88201904296875],
        'focal_length': [1145.5113525390625, 1144.77392578125],
        'radial_distortion': [-0.198384091258049, 0.21832367777824402, -0.008947807364165783],
        'tangential_distortion': [-0.0005872055771760643, -0.0018133620033040643],
        'res_w': 1000,
        'res_h': 1002,
        'azimuth': -110, # Only used for visualization
    },
]

h36m_cameras_extrinsic_params = {
    'S1': [
        {
            'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
            'rotation': [[-0.9153617321513369, 0.40180836633680234, 0.02574754463350265], 
                         [0.051548117060134555, 0.1803735689384521, -0.9822464900705729], 
                         [-0.399319034032262, -0.8977836111057917, -0.185819527201491]],
            'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
        },
        {
            'orientation': [0.6157187819480896, -0.764836311340332, -0.14833825826644897, 0.11794740706682205],
            'rotation': [[0.9281683400814921, 0.3721538354721445, 0.002248380248018696], 
                         [0.08166409428175585, -0.1977722953267526, -0.976840363061605], 
                         [-0.3630902204349604, 0.9068559102440475, -0.21395758897485287]], 
            'translation': [1761.278564453125, -5078.0068359375, 1606.2650146484375],
        },
        {
            'orientation': [0.14651472866535187, -0.14647851884365082, 0.7653023600578308, -0.6094175577163696],
            'rotation': [[-0.9141549520542256, -0.40277802228118775, -0.045722952682337906], 
                         [-0.04562341383935874, 0.21430849526487267, -0.9756999400261069], 
                         [0.4027893093720077, -0.889854894701693, -0.214287280609606]],
            'translation': [-1846.7777099609375, 5215.04638671875, 1491.972412109375],
        },
        {
            'orientation': [0.5834008455276489, -0.7853162288665771, 0.14548823237419128, -0.14749594032764435],
            'rotation': [[0.9141562410494211, -0.40060705854636447, 0.061905989962380774], 
                         [-0.05641000739510571, -0.2769531972942539, -0.9592261660183036], 
                         [0.40141783470104664, 0.8733904688919611, -0.2757767409202658]],
            'translation': [-1794.7896728515625, -3722.698974609375, 1574.8927001953125],
        },
    ],
    'S5': [
        {
            'orientation': [0.1467377245426178, -0.162370964884758, -0.7551892995834351, 0.6178938746452332],
            'rotation': [[-0.9042074184788829, 0.42657831374650107, 0.020973473936051274], 
                         [0.06390493744399675, 0.18368565260974637, -0.9809055713959477], 
                         [-0.4222855708380685, -0.8856017859436166, -0.1933503902128034]],
            'translation': [2097.3916015625, 4880.94482421875, 1605.732421875],
        },
        {
            'orientation': [0.6159758567810059, -0.7626792192459106, -0.15728192031383514, 0.1189815029501915],
            'rotation': [[0.9222116004775194, 0.38649075753002626, 0.012274293810989732], 
                         [0.09333184463870337, -0.19167233853095322, -0.9770111982052265], 
                         [-0.3752531555110883, 0.902156643264318, -0.21283434941998647]],
            'translation': [2031.7008056640625, -5167.93310546875, 1612.923095703125],
        },
        {
            'orientation': [0.14291371405124664, -0.12907841801643372, 0.7678384780883789, -0.6110143065452576],
            'rotation': [[-0.9258288614330635, -0.3728674116124112, -0.06173178026768599], 
                         [-0.023578112500148365, 0.220000562347259, -0.9752147584905696], 
                         [0.3772068291381898, -0.9014264506460582, -0.21247437993123308]],
            'translation': [-1620.5948486328125, 5171.65869140625, 1496.43701171875],
        },
        {
            'orientation': [0.5920479893684387, -0.7814217805862427, 0.1274748593568802, -0.15036417543888092],
            'rotation': [[0.9222815489764817, -0.3772688722588351, 0.0840532119677073], 
                         [-0.021177649402562934, -0.26645871124348197, -0.9636136478735888], 
                         [0.3859381447632816, 0.88694303832152, -0.25373962085111357]],
            'translation': [-1637.1737060546875, -3867.3173828125, 1547.033203125],
        },
    ],
    'S6': [
        {
            'orientation': [0.1337897777557373, -0.15692396461963654, -0.7571090459823608, 0.6198879480361938],
            'rotation': [[-0.9149503344107554, 0.4034864343564006, 0.008036345687245266], 
                         [0.07174776353922047, 0.1822275975157708, -0.9806351824867137], 
                         [-0.3971374371533952, -0.896655898321083, -0.19567845056940925]],
            'translation': [1935.4517822265625, 4950.24560546875, 1618.0838623046875],
        },
        {
            'orientation': [0.6147197484970093, -0.7628812789916992, -0.16174767911434174, 0.11819244921207428],
            'rotation': [[0.9197364689900042, 0.39209901596964664, 0.018525368698999664], 
                         [0.101478073351267, -0.19191459963948, -0.9761511087296542], 
                         [-0.37919260045353465, 0.899681692667386, -0.21630030892357308]],
            'translation': [1969.803955078125, -5128.73876953125, 1632.77880859375],
        },
        {
            'orientation': [0.1529948115348816, -0.13529130816459656, 0.7646096348762512, -0.6112781167030334],
            'rotation': [[-0.916577698818659, -0.39393483656788014, -0.06856140726771254], 
                         [-0.01984531630322392, 0.21607069980297702, -0.9761760169700323], 
                         [0.3993638509543854, -0.8933805444629346, -0.20586334624209834]],
            'translation': [-1769.596435546875, 5185.361328125, 1476.993408203125],
        },
        {
            'orientation': [0.5916101336479187, -0.7804774045944214, 0.12832270562648773, -0.1561593860387802],
            'rotation': [[0.9182950552949388, -0.3850769011116475, 0.09192372735651859], 
                         [-0.015534985886560007, -0.26706146429979655, -0.9635542737695438], 
                         [0.3955917790277871, 0.8833990913037544, -0.25122338635033875]],
            'translation': [-1721.668701171875, -3884.13134765625, 1540.4879150390625],
        },
    ],
    'S7': [
        {
            'orientation': [0.1435241848230362, -0.1631336808204651, -0.7548328638076782, 0.6188824772834778],
            'rotation': [[-0.9055764231419416, 0.42392653746206904, 0.014752378956221508], 
                         [0.06862812683752326, 0.18074371881263407, -0.9811329615890764], 
                         [-0.41859469903024304, -0.8874784498483331, -0.19277053457045695]],
            'translation': [1974.512939453125, 4926.3544921875, 1597.8326416015625],
        },
        {
            'orientation': [0.6141672730445862, -0.7638262510299683, -0.1596645563840866, 0.1177929937839508],
            'rotation': [[0.9212640765077017, 0.3886011826562522, 0.01617473877914905], 
                         [0.09922277503271489, -0.1946115441987536, -0.9758489574618522], 
                         [-0.3760682680727248, 0.9006194910741931, -0.21784671226815075]],
            'translation': [1937.0584716796875, -5119.7900390625, 1631.5665283203125],
        },
        {
            'orientation': [0.14550060033798218, -0.12874816358089447, 0.7660516500473022, -0.6127139329910278],
            'rotation': [[-0.9245069728829368, -0.37555597339631824, -0.06515034871105972], 
                         [-0.018955014220249346, 0.2160111098950734, -0.9762068980691586], 
                         [0.38069353097569036, -0.9012751584550871, -0.2068224461344045]],
            'translation': [-1741.8111572265625, 5208.24951171875, 1464.8245849609375],
        },
        {
            'orientation': [0.5912848114967346, -0.7821764349937439, 0.12445473670959473, -0.15196487307548523],
            'rotation': [[0.9228353966173104, -0.37440015452287667, 0.09055029013436408],
                         [-0.01498208436370467, -0.269786590656035, -0.9628035794752281],
                         [0.38490306298896904, 0.8871525910436372, -0.25457791009093983]],
            'translation': [-1734.7105712890625, -3832.42138671875, 1548.5830078125],
        },
    ],
    'S8': [
        {
            'orientation': [0.14110587537288666, -0.15589867532253265, -0.7561917304992676, 0.619644045829773],
            'rotation': [[-0.9115694669712032, 0.4106494283805017, 0.020202818036194434], 
                         [0.060907749548984036, 0.1834736632003901, -0.9811359034082424], 
                         [-0.40660958293025334, -0.8931430243150293, -0.19226072190306673]],
            'translation': [2150.65185546875, 4896.1611328125, 1611.9046630859375],
        },
        {
            'orientation': [0.6169601678848267, -0.7647668123245239, -0.14846350252628326, 0.11158157885074615],
            'rotation': [[0.9212640765077017, 0.3886011826562522, 0.01617473877914905], 
                         [0.09922277503271489, -0.1946115441987536, -0.9758489574618522], 
                         [-0.3760682680727248, 0.9006194910741931, -0.21784671226815075]],
            'translation': [2219.965576171875, -5148.453125, 1613.0440673828125],
        },
        {
            'orientation': [0.1471444070339203, -0.13377119600772858, 0.7670128345489502, -0.6100369691848755],
            'rotation': [[-0.9209075762929309, -0.3847355178017309, -0.0625125368875214], 
                         [-0.02568138180824641, 0.21992027027623712, -0.9751797482259595], 
                         [0.38893405939143305, -0.8964450100611084, -0.21240678280563546]],
            'translation': [-1571.2215576171875, 5137.0185546875, 1498.1761474609375],
        },
        {
            'orientation': [0.5927824378013611, -0.7825870513916016, 0.12147816270589828, -0.14631995558738708],
            'rotation': [[0.927667052235436, -0.3636062759574404, 0.08499597802942535], 
                         [-0.01666268768012713, -0.26770413351564454, -0.9633570738505596], 
                         [0.37303645269074087, 0.8922583555131325, -0.2543989622245125]],
            'translation': [-1476.913330078125, -3896.7412109375, 1547.97216796875],
        },
    ],
    'S9': [
        {
            'orientation': [0.15540587902069092, -0.15548215806484222, -0.7532095313072205, 0.6199594736099243],
            'rotation': [[-0.9033486204435297, 0.4269119782787646, 0.04132109321984796], 
                         [0.04153061098352977, 0.182951140059007, -0.9822444139329296], 
                         [-0.4268916470184284, -0.8855930460167476, -0.18299857527497945]],
            'translation': [2044.45849609375, 4935.1171875, 1481.2275390625],
        },
        {
            'orientation': [0.618784487247467, -0.7634735107421875, -0.14132238924503326, 0.11933968216180801],
            'rotation': [[0.9315720471487059, 0.36348288012373176, -0.007329176497134756], 
                         [0.06810069482701912, -0.19426747906725159, -0.9785818524481906], 
                         [-0.35712157080642226, 0.911120377575769, -0.20572758986325015]],
            'translation': [1990.959716796875, -5123.810546875, 1568.8048095703125],
        },
        {
            'orientation': [0.13357827067375183, -0.1367100477218628, 0.7689454555511475, -0.6100738644599915],
            'rotation': [[-0.9269344193869241, -0.3732303525241731, -0.03862235247246717], 
                         [-0.04725991098820678, 0.218240494552814, -0.9747500127472326], 
                         [0.37223525218497616, -0.901704048173249, -0.21993345934341726]],
            'translation': [-1670.9921875, 5211.98583984375, 1528.387939453125],
        },
        {
            'orientation': [0.5879399180412292, -0.7823407053947449, 0.1427614390850067, -0.14794869720935822],
            'rotation': [[0.9154607080837831, -0.39734606500700814, 0.06362229623477154], 
                         [-0.049406284684695274, -0.2678916756611978, -0.9621814117644814], 
                         [0.3993628813352506, 0.877695935238897, -0.26487569589663096]],
            'translation': [-1696.04345703125, -3827.099853515625, 1591.4127197265625],
        },
    ],
    'S11': [
        {
            'orientation': [0.15232472121715546, -0.15442320704460144, -0.7547563314437866, 0.6191070079803467],
            'rotation': [[-0.9059013006181885, 0.4217144115102914, 0.038727105014486805], 
                         [0.044493184429779696, 0.1857199061874203, -0.9815948619389944], 
                         [-0.4211450938543295, -0.8875049698848251, -0.1870073216538954]],
            'translation': [2098.440185546875, 4926.5546875, 1500.278564453125],
        },
        {
            'orientation': [0.6189449429512024, -0.7600917220115662, -0.15300633013248444, 0.1255258321762085],
            'rotation': [[0.9216646531492915, 0.3879848687925067, -0.0014172943441045224], 
                         [0.07721054863099915, -0.18699239961454955, -0.979322405373477], 
                         [-0.3802272982247548, 0.9024974149959955, -0.20230080971229314]],
            'translation': [2083.182373046875, -4912.1728515625, 1561.07861328125],
        },
        {
            'orientation': [0.14943228662014008, -0.15650227665901184, 0.7681233882904053, -0.6026304364204407],
            'rotation': [[-0.9063540572469627, -0.42053101768163204, -0.04093880896680188], 
                         [-0.0603212197838846, 0.22468715090881142, -0.9725620980997899], 
                         [0.4181909532208387, -0.8790161246439863, -0.2290130547809762]],
            'translation': [-1609.8153076171875, 5177.3359375, 1537.896728515625],
        },
        {
            'orientation': [0.5894251465797424, -0.7818877100944519, 0.13991211354732513, -0.14715361595153809],
            'rotation': [[0.91754082476548, -0.39226322025776267, 0.06517975852741943], 
                         [-0.04531905395586976, -0.26600517028098103, -0.9629057236990188], 
                         [0.395050652748768, 0.8805514269006645, -0.2618476013752581]],
            'translation': [-1590.738037109375, -3854.1689453125, 1578.017578125],
        },
    ],
}

def triangulation(pts, pmat):
    '''
    pts: (batch, njoints, nview, 2)
    pmat: (nview, 3, 4) or (batch, nview, 3, 4)
    '''

    dev = pts.device

    batch, njoint, nview = pts.shape[0:3]#(batch, njoints, nview, 2)
    print(batch)
    if len(pmat.shape) == 3:
        pmat = pmat.to(dev).view(1, nview, 3, 4).repeat(batch * njoint, 1, 1, 1) #(batch * njoints, nview, 3, 4)
    elif len(pmat.shape) == 4:
        pmat = pmat.to(dev).view(batch, 1, nview, 3, 4).repeat(1, njoint, 1, 1, 1).view(batch*njoint, nview, 3, 4) #(batch * njoints, nview, 3, 4)
    pts_compact = pts.view(batch*njoint, nview, 2, 1)
    A = pmat[:,:,2:3] * pts_compact #(batch*njoint, nview, 2, 4)
    A -= pmat[:,:,:2]
    A = A.view(-1, 2 * nview, 4)
    A_np = A.cpu().numpy()
    try:
        u, d, vt = np.linalg.svd(A_np)  # vt (batch*njoint, 4, 4)
        Xs = vt[:,-1,0:3]/vt[:,-1,3:]

    except np.linalg.LinAlgError:
        Xs = np.zeros((batch*njoint, 3), dtype=np.float32)
        #print("linalg")
    except FloatingPointError:
        # print(vt[:,-1,3:])
        div = vt[:,-1,3:]
        div[div==0] = float('inf')
        Xs = vt[:,-1,0:3]/vt[:,-1,3:]
        #print("FloatingPointError")

    # convert ndarr to tensor
    Xs = torch.as_tensor(Xs, dtype=torch.float32, device=dev)
    Xs = Xs.view(batch, njoint, 3)
    return Xs

def triangulation_torch(pts, pmat):
    '''
    pts: (batch, njoints, nview, 2)
    pmat: (nview, 3, 4) or (batch, nview, 3, 4)
    '''
    stats_sing_values = {"values":{}, "distances":{}}
    dev = pts.device

    batch, njoint, nview = pts.shape[0:3]#(batch, njoints, nview, 2)
    #print(batch)
    if len(pmat.shape) == 3:
        pmat = pmat.to(dev).view(1, nview, 3, 4).repeat(batch * njoint, 1, 1, 1) #(batch * njoints, nview, 3, 4)
    elif len(pmat.shape) == 4:
        pmat = pmat.to(dev).view(batch, 1, nview, 3, 4).repeat(1, njoint, 1, 1, 1).view(batch*njoint, nview, 3, 4) #(batch * njoints, nview, 3, 4)
    pts_compact = pts.view(batch*njoint, nview, 2, 1)
    A = pmat[:,:,2:3] * pts_compact #(batch*njoint, nview, 2, 4)
    A -= pmat[:,:,:2]
    A = A.view(-1, 2 * nview, 4)
    #A_np = A.cpu().numpy()
    # try:
    u, d, vt = torch.linalg.svd(A)  # vt (batch*njoint, 4, 4)
    #print(d.shape)
    #print("torch.min(d, dim=0) : "+str(torch.min(d, dim=0)))
    # stats_sing_values["values"]["min"] = torch.min(d)
    # stats_sing_values["values"]["max"] = torch.max(d)
    # stats_sing_values["values"]["mean"] = torch.mean(d)
    # stats_sing_values["values"]["std"] = torch.std(d)
    # d_1 = torch.unsqueeze(d, dim=-1).repeat(1, 1, 4)
    # d_2 = torch.transpose(d_1, -2, -1)
    #print(d_1[0])
    #print(d_2[0])
    #dist_sing_values = torch.abs(d_1-d_2)
    #print("torch.min(dist_sing_values, dim=0) : "+str(torch.min(dist_sing_values, dim=0)))
    # min_dist_sing_values = torch.min(dist_sing_values, dim=0)[0]
    # tri_sup_idxes = torch.triu_indices(min_dist_sing_values.shape[0], min_dist_sing_values.shape[1], offset=1)
    # min_dist_sing_values = min_dist_sing_values[tri_sup_idxes[0], tri_sup_idxes[1]]
    #print("min_dist_sing_values : "+str(min_dist_sing_values))
    # stats_sing_values["distances"]["min"]=torch.min(min_dist_sing_values)
    # max_dist_sing_values = torch.max(dist_sing_values, dim=0)[0]
    # max_dist_sing_values = max_dist_sing_values[tri_sup_idxes[0], tri_sup_idxes[1]]
    # stats_sing_values["distances"]["max"] = torch.max(max_dist_sing_values)
    # mean_dist_sing_values = torch.mean(dist_sing_values, dim=0)
    # mean_dist_sing_values = mean_dist_sing_values[tri_sup_idxes[0], tri_sup_idxes[1]]
    # stats_sing_values["distances"]["mean"] = torch.mean(mean_dist_sing_values)
    # std_dist_sing_values = torch.std(dist_sing_values, dim=0)
    # std_dist_sing_values = std_dist_sing_values[tri_sup_idxes[0], tri_sup_idxes[1]]
    # stats_sing_values["distances"]["std"] = torch.std(std_dist_sing_values)
    #input()
    Xs = vt[:,-1,0:3]/vt[:,-1,3:]

    # except np.linalg.LinAlgError:
    #     Xs = np.zeros((batch*njoint, 3), dtype=np.float32)
    #     #print("linalg")
    # except FloatingPointError:
    #     # print(vt[:,-1,3:])
    #     div = vt[:,-1,3:]
    #     div[div==0] = float('inf')
    #     Xs = vt[:,-1,0:3]/vt[:,-1,3:]
    #     #print("FloatingPointError")

    # convert ndarr to tensor
    # Xs = torch.as_tensor(Xs, dtype=torch.float32, device=dev)
    Xs = Xs.view(batch, njoint, 3)
    return Xs, stats_sing_values


def flip_data(batch_data):
    data_leaf = batch_data.detach()
    data_copy = copy.deepcopy(data_leaf)
    kps_right = [1, 2, 3, 14, 15, 16]
    kps_left = [4, 5, 6, 11, 12, 13]
    data_copy[:, :, :, 0] *= -1
    data_copy[:, :, kps_left + kps_right] = data_copy[:, :, kps_right + kps_left]
    return data_copy
def visulization(s, p3d, order=[0,2,1], name=''):
    plt.cla()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(p3d[s,0,:,order[0]].cpu(), p3d[s,0,:,order[1]].cpu(), p3d[s,0,:,order[2]].cpu(), marker='o')
    ax.plot3D(p3d[s,0,[0,7,8,9,10],order[0]].cpu(), p3d[s,0,[0,7,8,9,10],order[1]].cpu(), p3d[s,0,[0,7,8,9,10],order[2]].cpu(), 'red')
    ax.plot3D(p3d[s,0,[8,14,15,16],order[0]].cpu(), p3d[s,0,[8,14,15,16],order[1]].cpu(),p3d[s,0,[8,14,15,16],order[2]].cpu(), 'red')
    ax.plot3D(p3d[s,0,[8,11,12,13],order[0]].cpu(), p3d[s,0,[8,11,12,13],order[1]].cpu(),p3d[s,0,[8,11,12,13],order[2]].cpu(), 'red')
    ax.plot3D(p3d[s,0,[0,1,2,3],order[0]].cpu(), p3d[s,0,[0,1,2,3],order[1]].cpu(),p3d[s,0,[0,1,2,3],order[2]].cpu(), 'red')
    ax.plot3D(p3d[s,0,[0,4,5,6],order[0]].cpu(), p3d[s,0,[0,4,5,6],order[1]].cpu(),p3d[s,0,[0,4,5,6],order[2]].cpu(), 'red')
    plt.savefig('test_{}{}.png'.format(s, name))

class Human36mCamera(MocapDataset):
    def __init__(self, cfg): 
        super().__init__()
        self.cfg = cfg
        self._cameras = copy.deepcopy(h36m_cameras_extrinsic_params)
        self._skeleton = h36m_skeleton
        for subject, cameras in self._cameras.items():
            for i, cam in enumerate(cameras):
                cam.update(h36m_cameras_intrinsic_params[i])
                for k, v in cam.items():
                    if k not in ['id', 'res_w', 'res_h']:
                        cam[k] = np.array(v, dtype='float32')
                
                # Normalize camera frame
                cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'], h=cam['res_h']).astype('float32')
                cam['focal_length'] = cam['focal_length']/cam['res_w']*2
                if 'translation' in cam:
                    cam['translation'] = torch.from_numpy(cam['translation'])/1000 # mm to meters
                if 'rotation' in cam:
                    cam['rotation'] = torch.FloatTensor(cam['rotation'])
                    
                # Add intrinsic parameters vector
                
                cam['project_to_2d_linear'] = torch.from_numpy(np.concatenate((cam['focal_length'],
                                                   cam['center'],
                                                   cam['radial_distortion'],
                                                   cam['tangential_distortion'])))
        wd3d_path = './data/data_3d_h36m.npz'
        data_mhf = np.load(wd3d_path, allow_pickle=True)['positions_3d'].item()
        self._data = {}
        for subject, actions in data_mhf.items():
            self._data[subject] = {}
            for action_name, positions in actions.items():
                self._data[subject][action_name] = {
                    'positions': positions,
                    'cameras': self._cameras[subject],
                }
        self.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])

        self.camera_set = {}
        for subject in self._cameras.keys():
            print('***', subject)
            self.camera_set[subject] = {}
            cameras = self._cameras[subject]
            N_cam = len(cameras)
            temp_cam = None
            temp_r = None
            temp_t = None
            for cam in range(N_cam):
                camera = cameras[cam]
                in_param = camera['project_to_2d_linear'].view(1, -1)
                r = camera['rotation'].view(1, 3, 3)
                t = camera['translation'].view(1, 3)
                if temp_cam is None:
                    temp_cam = in_param
                    temp_r = r
                    temp_t = t
                else:
                    temp_cam = torch.cat((temp_cam, in_param), dim = 0)
                    temp_r = torch.cat((temp_r, r), dim = 0)
                    temp_t = torch.cat((temp_t, t), dim = 0)
            temp_cam = temp_cam.view(1, N_cam, 1, -1)
            # print(temp_r.shape)
            # print(temp_t.shape)
            # input()
            #####world->camera
            exi_mat = torch.zeros(N_cam, 3, 4)
            exi_mat[:,:,:3] = temp_r
            exi_mat[:,:,-1] = torch.einsum('njc,nkc->nk',  -temp_t.view(4, 1, 3), temp_r) 
            self.camera_set[subject]['exi_mat'] = exi_mat
            #####camera->world
            exi_mat_inv = torch.zeros(N_cam, 3, 4)
            exi_mat_inv[:,:,:3] = temp_r.permute(0, 2, 1)
            exi_mat_inv[:,:,-1] = temp_t.view(4,3)
            self.camera_set[subject]['exi_mat_inv'] = exi_mat_inv

            #####camera->image
            K = torch.zeros(N_cam, 3, 3)
            K[:, 0, 0] = temp_cam[0,:,0,0]
            K[:,1,1] = temp_cam[0,:,0,1]
            K[:,0,2] = temp_cam[0,:,0,2]
            K[:,1,2] = temp_cam[0,:,0,3]
            K[:,2,2] = 1
            self.camera_set[subject]['K'] = K
            self.camera_set[subject]['intrin'] = temp_cam.squeeze(0).squeeze(1)
            ######world->image
            prj_mat = torch.einsum('nkj,njc->nkc', K, exi_mat)#(N_view, 3, 4)
            self.camera_set[subject]['prj_mat'] = prj_mat

    def remove_joints(self, joints_to_remove):
        kept_joints = self._skeleton.remove_joints(joints_to_remove)
        for subject in self._data.keys():
            for action in self._data[subject].keys():
                s = self._data[subject][action]
                s['positions'] = s['positions'][:, kept_joints]

    def p2d_cam3d(self, p2d, subject, view_list, debug=False):
        '''
        p2d: (B, T,J, C, N) 
        '''
        B, T, J, C, N = p2d.shape
        p2d = p2d.permute(0, 1, 2, 4, 3).contiguous() #(B, T, J, N, C)
        p2d = p2d.view(B*T, J, N, C)
        prj_mat = self.camera_set[subject]['prj_mat']
        exi_mat = self.camera_set[subject]['exi_mat']

        trj_w3d = triangulation(p2d, prj_mat[view_list,...]) #(B*T, J, 3)

        trj_w3d_homo = torch.cat((trj_w3d, torch.ones(trj_w3d.shape[0], 17, 1)), dim = -1)
        trj_c3d = torch.einsum('nkc,mjc->mnjk', exi_mat[view_list, ...], trj_w3d_homo) #(B*T, N, J, 3)
        if debug==True:
            exi_mat_inv = self.camera_set[subject]['exi_mat_inv'][view_list, ...] #(N, 3, 4)
        trj_c3d = trj_c3d.view(B, T, N, J, 3).permute(0, 1, 3, 4, 2).contiguous() ##B, T, J, 3, N)
        
        trj_c3d = trj_c3d - trj_c3d[:,:,:1]
        if debug==False:
            return trj_c3d, trj_c3d
        else:
            return trj_c3d, trj_w3d

    def p2d_cam3d_batch(self, p2d, subject_list, view_list, debug=False, extri=None, proj=None, is_predicted_params=True):
        '''
        p2d: (B, T,J, C, N)
        '''
        B, T, J, C, N = p2d.shape
        #trj_c3d = torch.zeros((B, T, J, 3, N))
        if self.cfg.TRAIN.LEARN_CAM_PARM:
            prj_mat=[]
            exi_mat=[]
        else:
            prj_mat = torch.zeros(B, 4, 3, 4)
            exi_mat = torch.zeros(B, 4, 3, 4)
        for inx, sub in enumerate(subject_list):
            if self.cfg.TRAIN.LEARN_CAM_PARM:
                if is_predicted_params:
                    exi_mat.append(torch.unsqueeze(extri[inx].to(p2d.device), dim=0))
                    prj_mat.append(torch.unsqueeze(proj[inx].to(p2d.device), dim=0))
                else:
                    exi_mat.append(torch.unsqueeze(extri[sub[0]], dim=0))
                    prj_mat.append(torch.unsqueeze(proj[sub[0]], dim=0))
            else:
                prj_mat[inx] = self.camera_set[sub[0]]['prj_mat']
                exi_mat[inx] = self.camera_set[sub[0]]['exi_mat']
        if self.cfg.TRAIN.LEARN_CAM_PARM:
            prj_mat = torch.cat(prj_mat, dim=0)
            exi_mat = torch.cat(exi_mat, dim=0)
        trj_w3d, stats_sing_values = triangulation_torch(p2d.view(-1, J, C, N).permute(0, 1, 3, 2).contiguous(), prj_mat[:, view_list,...]) #(B*T, J, 3)
        trj_w3d_homo = torch.cat((trj_w3d, torch.ones(trj_w3d.shape[0], 17, 1).to(p2d.device)), dim = -1)
        trj_c3d = torch.einsum('mnkc,mjc->mnjk', exi_mat[:,view_list, ...], trj_w3d_homo) #(B*T, N, J, 3)
        trj_c3d = trj_c3d.view(B, T, N, J, 3).permute(0, 1, 3, 4, 2).contiguous() ##B, T, J, 3, N)
        trj_c3d = trj_c3d - trj_c3d[:,:,:1]
        return trj_c3d, stats_sing_values, trj_w3d

    def p2d_cam3d_batch_with_root(self, p2d, subject_list, view_list, debug=False, extri=None, proj=None, is_predicted_params=True):
        '''
        p2d: (B, T,J, C, N)
        '''
        B, T, J, C, N = p2d.shape
        #trj_c3d = torch.zeros((B, T, J, 3, N))
        if self.cfg.TRAIN.LEARN_CAM_PARM:
            prj_mat=[]
            exi_mat=[]
        else:
            prj_mat = torch.zeros(B, 4, 3, 4)
            exi_mat = torch.zeros(B, 4, 3, 4)
        for inx, sub in enumerate(subject_list):
            if self.cfg.TRAIN.LEARN_CAM_PARM:
                if is_predicted_params:
                    exi_mat.append(torch.unsqueeze(extri[inx].to(p2d.device), dim=0))
                    prj_mat.append(torch.unsqueeze(proj[inx].to(p2d.device), dim=0))
                else:
                    exi_mat.append(torch.unsqueeze(extri[sub[0]], dim=0))
                    prj_mat.append(torch.unsqueeze(proj[sub[0]], dim=0))
            else:
                prj_mat[inx] = self.camera_set[sub[0]]['prj_mat']
                exi_mat[inx] = self.camera_set[sub[0]]['exi_mat']
        if self.cfg.TRAIN.LEARN_CAM_PARM:
            prj_mat = torch.cat(prj_mat, dim=0)
            exi_mat = torch.cat(exi_mat, dim=0)
        trj_w3d, stats_sing_values = triangulation_torch(p2d.view(-1, J, C, N).permute(0, 1, 3, 2).contiguous(), prj_mat[:, view_list,...]) #(B*T, J, 3)

        # if self.cfg.TRAIN.PREDICT_ROOT:
        #     root_pred = trj_w3d[:, :1]
        #     root_pred_clipped = torch.maximum(torch.minimum(root_pred, torch.ones_like(root_pred) * 10),  torch.ones_like(root_pred) * (-10))
        #     trj_w3d = torch.maximum(torch.minimum(trj_w3d-root_pred, torch.ones_like(trj_w3d) * 1.5),  torch.ones_like(trj_w3d) * (-1.5))+root_pred_clipped

        trj_w3d_homo = torch.cat((trj_w3d, torch.ones(trj_w3d.shape[0], 17, 1).to(p2d.device)), dim = -1)
        trj_c3d = torch.einsum('mnkc,mjc->mnjk', exi_mat[:,view_list, ...], trj_w3d_homo) #(B*T, N, J, 3)
        trj_c3d = trj_c3d.view(B, T, N, J, 3).permute(0, 1, 3, 4, 2).contiguous() ##B, T, J, 3, N)
        #trj_c3d = trj_c3d - trj_c3d[:,:,:1]
        return trj_c3d, stats_sing_values, trj_w3d

    def p3dcam_3dwd_batch(self, p3d, subject, view_list, extri_inv=None, is_predicted_params=True):
        #p3d: 3d poses in camera space
        #p_3dwd: returned 3d poses in world space
        p3d = p3d[..., self.cfg.H36M_DATA.TRAIN_CAMERAS]
        p3d = p3d.permute(0, 4, 1, 2, 3)
        B, N, T, J, C = p3d.shape
        #extri_mat = torch.zeros(B, 4, 3, 4).unsqueeze(2).unsqueeze(3).permute(0,1,2,3,5,4)
        if self.cfg.TRAIN.LEARN_CAM_PARM:
            exi_mat_inv = []
        else:
            exi_mat_inv =torch.zeros(B, len(view_list), 3, 4)
        for inx, sub in enumerate(subject):
            #print('Calling the {} exi_mat_inv of subject {}, {}'.format(inx, sub[0], sub[1]))
            #self.camera_set[sub[0]]['exi_mat_inv'].shape: torch.Size([4, 3, 4])
            if self.cfg.TRAIN.LEARN_CAM_PARM:
                if is_predicted_params:
                    exi_mat_inv.append(torch.unsqueeze(extri_inv[inx], dim=0))
                else:
                    exi_mat_inv.append(torch.unsqueeze(extri_inv[sub[0]], dim=0))
            else:
                exi_mat_inv[inx] = self.camera_set[sub[0]]['exi_mat_inv'][view_list]
        if self.cfg.TRAIN.LEARN_CAM_PARM:
            exi_mat_inv = torch.cat(exi_mat_inv, dim=0)
        #print(exi_mat_inv.shape)
        #len(extri_mat)=2; extri_mat[0].shape: torch.Size([4, 1, 1, 4, 3])
        #p3d.shape: torch.Size([2, 4, 1, 17, 3])
        p_3dwd_homo = torch.cat((p3d.squeeze(dim=2), torch.ones(p3d.shape[0], N, J, 1).to(p3d.device)), dim = -1)
        p_3dwd = torch.einsum('tnqc,tnjc->tnjq', exi_mat_inv.to(p3d.device), p_3dwd_homo)

        #p_3dwd_homo.shape: torch.Size([2, 4, 1, 17, 4])
        #p_3dwd.shape: torch.Size([2, 4, 1, 17, 3])
        #np.save('p_3dwd.npy', p_3dwd.to('cpu').detach().numpy())
        #fig = plt.figure()
        #ax = fig.add_subplot(projection='3d')
        #xs = p_3dwd[0,0,0,:,0].to('cpu').detach().numpy()
        #ys = p_3dwd[0,0,0,:,1].to('cpu').detach().numpy()
        #zs = p_3dwd[0,0,0,:,2].to('cpu').detach().numpy()
        #ax.scatter(xs, ys, zs, marker='o')
        #ax.set_xlabel('X Label')
        #ax.set_ylabel('Y Label')
        #ax.set_zlabel('Z Label')
        #plt.savefig('pos_3d.png')
        #plt.show()
        return p_3dwd

    def p2d_world3d_batch(self, p2d, subject_list, view_list, debug=False, extri=None, proj=None, is_predicted_params=True):
        '''
        p2d: (B, T,J, C, N)
        '''
        B, T, J, C, N = p2d.shape
        #trj_c3d = torch.zeros((B, T, J, 3, N))
        if self.cfg.TRAIN.LEARN_CAM_PARM:
            prj_mat=[]
            exi_mat=[]
        else:
            prj_mat = torch.zeros(B, 4, 3, 4)
            exi_mat = torch.zeros(B, 4, 3, 4)
        for inx, sub in enumerate(subject_list):
            if self.cfg.TRAIN.LEARN_CAM_PARM:
                if is_predicted_params:
                    exi_mat.append(torch.unsqueeze(extri[inx].to(p2d.device), dim=0))
                    prj_mat.append(torch.unsqueeze(proj[inx].to(p2d.device), dim=0))
                else:
                    exi_mat.append(torch.unsqueeze(extri[sub[0]], dim=0))
                    prj_mat.append(torch.unsqueeze(proj[sub[0]], dim=0))
            else:
                prj_mat[inx] = self.camera_set[sub[0]]['prj_mat']
                exi_mat[inx] = self.camera_set[sub[0]]['exi_mat']
        if self.cfg.TRAIN.LEARN_CAM_PARM:
            prj_mat = torch.cat(prj_mat, dim=0)
            exi_mat = torch.cat(exi_mat, dim=0)
        trj_w3d = triangulation_torch(p2d.view(-1, J, C, N).permute(0, 1, 3, 2).contiguous(), prj_mat[:, view_list,...]) #(B*T, J, 3)
        trj_w3d_homo = torch.cat((trj_w3d, torch.ones(trj_w3d.shape[0], 17, 1).to(p2d.device)), dim = -1)
        return trj_w3d_homo

    def cam3d_world3d_batch(self, trj_w3d, subject_list, view_list):
        B, T, J, C, N = trj_w3d.shape
        prj_mat = torch.zeros(B, 4, 3, 4)
        exi_mat = torch.zeros(B, 4, 3, 4)
        ref_cam_to_other_cams= torch.zeros(B, 4, 3, 4)
        for inx, sub in enumerate(subject_list):
            prj_mat[inx] = self.camera_set[sub[0]]['prj_mat']
            ref_cam_to_h36m_world=self.camera_set[sub[0]]['exi_mat_inv'][-1].repeat(N, 1, 1)
            exi_mat[inx] = self.camera_set[sub[0]]['exi_mat']
            ref_cam_to_other_cams[inx] = torch.einsum('nkj,njc->nkc', ref_cam_to_h36m_world, exi_mat[inx])  # (N_view, 3, 4)
        #ToDo get exi_mat_inv for last camera and compose exi_matrices with it!
        trj_w3d_homo = torch.cat((trj_w3d, torch.ones(trj_w3d.shape[0], 17, 1).to(trj_w3d.device)), dim = -1)
        trj_c3d = torch.einsum('mnkc,mjc->mnjk', ref_cam_to_other_cams[:, view_list, ...], trj_w3d_homo)  # (B*T, N, J, 3)
        trj_c3d = trj_c3d.view(B, T, N, J, 3).permute(0, 1, 3, 4, 2).contiguous()  ##B, T, J, 3, N)
        trj_c3d = trj_c3d - trj_c3d[:, :, :1]
        return trj_c3d


    """def p2d_world3d_batch(self, p2d, subject_list, view_list):
        '''
        p2d: (B, T, J, C, N)
        '''
        B, T, J, C, N = p2d.shape
        p2d=p2d[...,:4].permute(0, 1, 2, 4, 3).view(B*T, J, -1, C).contiguous() #(B, T, J, N, C)
        prj_mat = torch.zeros(B, 4, 3, 4)
        for inx, sub in enumerate(subject_list):
            prj_mat[inx] = self.camera_set[sub[0]]['prj_mat']
        prj_mat=prj_mat.repeat(T, 1, 1, 1)
        trj_w3d = triangulation(p2d, prj_mat) #(B*T, J, 3)
        trj_w3d = trj_w3d.view(B, T, J, 3).contiguous()

        return trj_w3d"""


    def p3d_im2d(self, p3d, subject, view_list):
        '''
        :param p3d:
        :param subject:
        :param view_list:
        :return: im2d
        '''
        p3d = p3d[..., self.cfg.H36M_DATA.TRAIN_CAMERAS]
        p3d = p3d.permute(0, 4, 1, 2, 3)
        B, N, T, J, C = p3d.shape
        p_2d = torch.zeros((B, N, T, J, 2))
        for inx, sub, p3d_ in zip(range(B), subject, p3d):
            intri_mat = self.camera_set[sub[0]]['intrin']
            intri_mat = intri_mat.to(p3d_.device)
            p_2d[inx,:,:,:,:] = project_to_2d(p3d_, intri_mat)
        return p_2d.permute(0,2,3,4,1), p_2d
    
    def p3d_im2d_batch(self, p3d, subject, view_list, with_distor=False, flip=[], gt_2d=None):
        '''
        :param p3d:
        :param subject:
        :param view_list:
        :return: im2d
        '''
        flip_idx = [idx for idx, ele in enumerate(flip) if ele == True]
        unflip_idx = [idx for idx, ele in enumerate(flip) if ele == False]
        while len(p3d.shape)!=5:
            p3d=p3d.unsqueeze(1)
        p3d = p3d[..., self.cfg.H36M_DATA.TRAIN_CAMERAS]
        B, T, J, C, N = p3d.shape
        if not with_distor:
            K = self.camera_set[subject[0][0]]['K'][view_list]
            p3d = p3d.squeeze(1).view((-1, C, N)).permute(0, 2, 1).unsqueeze(3).contiguous()
            K = K.unsqueeze(0).repeat(B*J, 1, 1, 1).to(p3d.device)
            p_2d = torch.einsum('bvik,bvkj->bvij', K, p3d).view(B, J, N, 3, 1).squeeze().contiguous()
            p_2d = (p_2d/p_2d[:,:,:,-1].unsqueeze(3))[:,:,:,:2]
        else:
            p3d = p3d.permute(0, 4, 1, 2, 3).contiguous()
            p3d_copy = p3d.detach()
            p3d_flip = p3d[flip_idx]
            p3d = p3d.contiguous().view((-1, T, J, C)).contiguous()
            p3d_flip = p3d_flip.contiguous().view((-1, T, J, C)).contiguous()
            intri_mat_ori = self.camera_set[subject[0][0]]['intrin'][view_list]   # [4, 9]
            intri_mat = intri_mat_ori.detach().repeat([B, 1, 1]).view((-1, 9)).contiguous().to(p3d.device)  # [B*4, 9]
            p_2d = project_to_2d(p3d, intri_mat)
            p_2d = p_2d.unsqueeze(0).view((B, N, T, J, 2)).contiguous()
            '''
            intri_mat_flip = intri_mat_ori
            intri_mat_flip[:,0] *= -1
            intri_mat_flip = intri_mat_flip.repeat([len(flip_idx), 1, 1]).view((-1, 9)).contiguous().to(p3d.device)
            p_2d_flip = project_to_2d(p3d_flip, intri_mat_flip)
            p_2d_flip = p_2d_flip.unsqueeze(0).view((len(flip_idx), N, T, J, 2)).contiguous()
            '''
            if len(flip_idx)!=0:
                intri_mat_flip = intri_mat_ori.repeat([len(flip_idx), 1, 1]).view((-1, 9)).contiguous().to(p3d.device)
                p3d_flip_back = flip_data(p3d_flip)
                p2d_flip_back = project_to_2d(p3d_flip_back, intri_mat_flip)
                p2d_flip = flip_data(p2d_flip_back)
                p2d_flip = p2d_flip.unsqueeze(0).view((len(flip_idx), N, T, J, 2)).contiguous()
                if gt_2d is not None:
                    gt_2d_flip = gt_2d[flip_idx]
                    gt_2d_unflip = gt_2d[unflip_idx]
                    p_2d_unflip = p_2d[unflip_idx]
                    flip1 = flip_data(p3d_flip)
                    flip_flip = flip_data(flip1)
                    visulization(0, p3d_flip)
                    visulization(0, flip1.cpu(), name='flip')
                    visulization(0, flip_flip.cpu(), name='flip_flip')
                    print("Fliped loss is {}; Unfliped loss is {}".format(mpjpe(p2d_flip.permute(0,2,3,4,1), gt_2d_flip), mpjpe(p_2d_unflip.permute(0,2,3,4,1),gt_2d_unflip)))
                p_2d[flip_idx] = p2d_flip
                if gt_2d is not None:
                    print(mpjpe(p_2d.permute(0,2,3,4,1), gt_2d))
        return p_2d

    def recover_quaternion_to_extri(self, theta, t):
        """
        Covert a quaternion into a full three-dimensional rotation matrix.

        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix.
                 This rotation matrix converts a point in the local reference
                 frame to a point in the global reference frame.
        """
        N_sub = theta.shape[0]
        N_cam = theta.shape[1]
        # Extract the values from Q
        theta = torch.nn.functional.normalize(theta, dim=-1)
        #print(theta.shape)
        #print(torch.linalg.norm(theta, dim=-1).shape)
        if not torch.prod(torch.isclose(torch.linalg.norm(theta, dim=-1), torch.ones(N_sub, N_cam).double().cuda())):
            #raise RuntimeError('Action contained non unit quaternion!')
            print("!!!!!!!!!!!!!!!!!!! non unit quaternion found !!!!!!!!!!!!!!!!!!!")
        q0 = theta[:, :, 0:1]
        q1 = theta[:, :, 1:2]
        q2 = theta[:, :, 2:3]
        q3 = theta[:, :, 3:4]


        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)

        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)

        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1

        # 3x3 rotation matrix
        #rot_matrix = torch.tensor([[r00, r01, r02],
                               #[r10, r11, r12],
                               #[r20, r21, r22]])
        l_1 = torch.unsqueeze(torch.cat((r00, r01, r02),2), -2)
        l_2 = torch.unsqueeze(torch.cat((r10, r11, r12), 2), -2)
        l_3 = torch.unsqueeze(torch.cat((r20, r21, r22), 2), -2)
        rot_matrix = torch.cat([l_1, l_2, l_3], dim=-2)



        T = torch.zeros([N_sub, N_cam, 4, 4]).to(t.device)
        T[:, :, 0, 3] = t[:, :, 0]
        T[:, :, 1, 3] = t[:, :, 1]
        T[:, :, 2, 3] = t[:, :, 2]
        T[:, :, 3, 3] = 1.0  # 4x4
        # print("T[0,0] : " + str(T[0, 0]))
        T = T[:, :, :3, -1]
        # print("T[0,0] : "+str(T[0,0]))

        return rot_matrix, T

    def recover_theta_to_extri(self, theta, t):
        N_sub = theta.shape[0]
        N_cam = theta.shape[1]
        Rx = torch.zeros([N_sub, N_cam, 4, 4])
        Ry = torch.zeros([N_sub, N_cam, 4, 4])
        Rz = torch.zeros([N_sub, N_cam, 4, 4])
        T = torch.zeros([N_sub, N_cam, 4, 4])

        # Rx[:, :, 0, 0] = 1.0
        # Rx[:, :, 1, 1] = torch.cos(torch.tanh(theta[:, :, 0]) * math.pi)  # cos(theta)
        # Rx[:, :, 1, 2] = -torch.sin(torch.tanh(theta[:, :, 0]) * math.pi)  # -sin(theta)
        # Rx[:, :, 2, 1] = torch.sin(torch.tanh(theta[:, :, 0]) * math.pi)  # sin(theta)
        # Rx[:, :, 2, 2] = torch.cos(torch.tanh(theta[:, :, 0]) * math.pi)  # cos(theta)
        # Rx[:, :, 3, 3] = 1.0
        # # print("Rx[0,0] : "+str(Rx[0,0]))
        #
        # Ry[:, :, 0, 0] = torch.cos(torch.tanh(theta[:, :, 1]) * math.pi)  # cos(theta)
        # Ry[:, :, 0, 2] = torch.sin(torch.tanh(theta[:, :, 1]) * math.pi)  # sin(theta)
        # Ry[:, :, 1, 1] = 1.0
        # Ry[:, :, 2, 0] = -torch.sin(torch.tanh(theta[:, :, 1]) * math.pi)  # -sin(theta)
        # Ry[:, :, 2, 2] = torch.cos(torch.tanh(theta[:, :, 1]) * math.pi)  # cos(theta)
        # Ry[:, :, 3, 3] = 1.0
        # # print("Ry[0,0] : " + str(Ry[0, 0]))
        #
        # Rz[:, :, 0, 0] = torch.cos(torch.tanh(theta[:, :, 2]) * math.pi)  # cos(theta)
        # Rz[:, :, 0, 1] = -torch.sin(torch.tanh(theta[:, :, 2]) * math.pi)  # -sin(theta)
        # Rz[:, :, 1, 0] = torch.sin(torch.tanh(theta[:, :, 2]) * math.pi)  # sin(theta)
        # Rz[:, :, 1, 1] = torch.cos(torch.tanh(theta[:, :, 2]) * math.pi)  # cos(theta)
        # Rz[:, :, 2, 2] = 1.0
        # Rz[:, :, 3, 3] = 1.0
        Rx[:, :, 0, 0] = 1.0
        Rx[:, :, 1, 1] = torch.cos(theta[:, :, 0])  # cos(theta)
        Rx[:, :, 1, 2] = -torch.sin(theta[:, :, 0])  # -sin(theta)
        Rx[:, :, 2, 1] = torch.sin(theta[:, :, 0])  # sin(theta)
        Rx[:, :, 2, 2] = torch.cos(theta[:, :, 0])  # cos(theta)
        Rx[:, :, 3, 3] = 1.0
        # print("Rx[0,0] : "+str(Rx[0,0]))

        Ry[:, :, 0, 0] = torch.cos(theta[:, :, 1])  # cos(theta)
        Ry[:, :, 0, 2] = torch.sin(theta[:, :, 1])  # sin(theta)
        Ry[:, :, 1, 1] = 1.0
        Ry[:, :, 2, 0] = -torch.sin(theta[:, :, 1])  # -sin(theta)
        Ry[:, :, 2, 2] = torch.cos(theta[:, :, 1])  # cos(theta)
        Ry[:, :, 3, 3] = 1.0
        # print("Ry[0,0] : " + str(Ry[0, 0]))

        Rz[:, :, 0, 0] = torch.cos(theta[:, :, 2])  # cos(theta)
        Rz[:, :, 0, 1] = -torch.sin(theta[:, :, 2])  # -sin(theta)
        Rz[:, :, 1, 0] = torch.sin(theta[:, :, 2])  # sin(theta)
        Rz[:, :, 1, 1] = torch.cos(theta[:, :, 2])  # cos(theta)
        Rz[:, :, 2, 2] = 1.0
        Rz[:, :, 3, 3] = 1.0
        #print("Rz[0,0] : " + str(Rz[0, 0]))

        R = torch.einsum('scij,scjk->scik', Rx, Ry)
        R = torch.einsum('scij,scjk->scik', R, Rz)
        #print(R[0, 0])
        R = R[:, :, :3, :3]
        #print("R[0,0] : " + str(R[0, 0]))

        # T[:, :, 0, 3] = torch.sigmoid(t[:, :, 0])*12-6
        # T[:, :, 1, 3] = torch.sigmoid(t[:, :, 1])*12-6
        # T[:, :, 2, 3] = torch.sigmoid(t[:, :, 2])*12-6
        # T[:, :, 0, 3] = torch.tanh(t[:, :, 0])*10
        # T[:, :, 1, 3] = torch.tanh(t[:, :, 1])*10
        # T[:, :, 2, 3] = torch.tanh(t[:, :, 2])*1
        T[:, :, 0, 3] = t[:, :, 0]
        T[:, :, 1, 3] = t[:, :, 1]
        T[:, :, 2, 3] = t[:, :, 2]
        T[:, :, 3, 3] = 1.0  # 4x4
        #print("T[0,0] : " + str(T[0, 0]))
        T = T[:, :, :3, -1]
        #print("T[0,0] : "+str(T[0,0]))

        return R, T

    """def recover_extri_extri_inv_predicted_params(self, params, subjects, reduced_params=False):
        params = torch.mean(params, dim=1) # (B, 12, N)
        B, _, N = params.shape
        # # print("params.shape : "+str(params.shape))
        # params = params.view(B, 3, -1, N)
        if reduced_params:
            learn_theta = params[:, :4, :].permute(0, 2, 1).contiguous()
            # print("learn_theta.shape : " + str(learn_theta.shape))
            learn_t = params[:, 4:, :].permute(0, 2, 1).contiguous()
            # print("learn_t.shape : " + str(learn_theta.shape))
            # input()
            #learn_r, learn_t = self.recover_theta_to_extri(learn_theta, learn_t)
            learn_r, learn_t = self.recover_quaternion_to_extri(learn_theta, learn_t)
        else:
            learn_r = params.view(B, 3, -1, N)[:, :, :3].permute(0, 3, 1, 2).contiguous()
            learn_t = params.view(B, 3, -1, N)[:, :, 3:].permute(0, 3, 1, 2).contiguous()
        #learn_t[:, :-1, -1, -1]= torch.tanh(learn_t[:, :-1, -1, -1])*0.5
        idx = 0
        all_extri = []
        all_extri_inv = []
        all_proj = []
        loss_learn_t = []
        #print(subjects[0])
        for subject in subjects:
            self.camera_set[subject[0]] = {}
            cameras = self._cameras[subject[0]]
            N_cam = N#len(cameras)
            temp_cam = None
            temp_r = None
            temp_t = None
            for cam in range(N_cam):
                camera = cameras[cam]
                in_param = camera['project_to_2d_linear'].view(1, -1)
                r = camera['rotation'].view(1, 3, 3)
                t = camera['translation'].view(1, 3)
                if temp_cam is None:
                    temp_cam = in_param
                    temp_r = r
                    temp_t = t
                else:
                    temp_cam = torch.cat((temp_cam, in_param), dim=0)
                    temp_r = torch.cat((temp_r, r), dim=0)
                    temp_t = torch.cat((temp_t, t), dim=0)
            temp_cam = temp_cam.view(1, N_cam, 1, -1)

            # cameras = self._cameras[subject]
            N_cam = N
            #####world->camera
            exi_mat = torch.zeros(N_cam, 3, 4)
            # print(learn_r[idx].shape)
            exi_mat[:, :, :3] = learn_r[idx]

            exi_mat[:, :, -1] = torch.einsum('njc,nkc->nk', -learn_t[idx].view(N, 1, 3), learn_r[idx])
            # ref = temp_t[:1].repeat(4, 1)
            # #temp_t_ = temp_t.clone()
            # temp_t = temp_t-ref
            # #print(temp_t)
            # exi_mat[:, :, -1] = torch.einsum('njc,nkc->nk', -temp_t.to(learn_r.device).view(N, 1, 3), learn_r[idx])
            # print(exi_mat[:, :, -1])
            # input()
            #
            # fig = plt.figure("cam 0 ref")
            # ax = fig.add_subplot(projection='3d')
            #
            # x_gt = temp_t[:, 0].cpu().numpy()
            # y_gt = temp_t[:, 1].cpu().numpy()
            # z_gt = temp_t[:, 2].cpu().numpy()
            # ax.scatter(x_gt, y_gt, z_gt, marker='o', color='r', label="gt 3D")
            #
            # fig = plt.figure("default ref")
            # ax = fig.add_subplot(projection='3d')
            # x_gt = temp_t_[:, 0].cpu().numpy()
            # y_gt = temp_t_[:, 1].cpu().numpy()
            # z_gt = temp_t_[:, 2].cpu().numpy()
            # ax.scatter(x_gt, y_gt, z_gt, marker='o', color='r', label="gt 3D")
            # plt.show()
            #exi_mat[-1, :, :]=torch.cat([torch.eye(3, 3), torch.zeros(3, 1)], dim=-1)
            all_extri.append(exi_mat)
            #####camera->world
            exi_mat_inv = torch.zeros(N_cam, 3, 4)
            exi_mat_inv[:, :, :3] = learn_r[idx].permute(0, 2, 1).contiguous()
            exi_mat_inv[:, :, -1] = learn_t[idx].view(N, 3)
            #exi_mat_inv[:, :, -1] = temp_t.to(learn_r[idx].device).view(4, 3)
            #exi_mat_inv[-1,:,:]=torch.cat([torch.eye(3, 3), torch.zeros(3, 1)], dim=-1)
            all_extri_inv.append(exi_mat_inv)
            #####camera->image
            K = torch.zeros(N_cam, 3, 3)
            K[:, 0, 0] = temp_cam[0, :, 0, 0]
            K[:, 1, 1] = temp_cam[0, :, 0, 1]
            K[:, 0, 2] = temp_cam[0, :, 0, 2]
            K[:, 1, 2] = temp_cam[0, :, 0, 3]
            K[:, 2, 2] = 1
            self.camera_set[subject[0]]['K'] = K
            self.camera_set[subject[0]]['intrin'] = temp_cam.squeeze(0).squeeze(1)
            ######world->image
            prj_mat = torch.einsum('nkj,njc->nkc', K, exi_mat)  # (N_view, 3, 4)
            all_proj.append(prj_mat)
            idx = idx + 1
        return all_extri, all_extri_inv, all_proj"""

    def recover_extri_extri_inv_predicted_params(self, params, subjects, reduced_params=False):
        """
        Args:
            params: B, T, 12, N
        Returns:
        """
        params = torch.mean(params, dim=1) # (B, 12, N)
        B, _, N = params.shape
        # # print("params.shape : "+str(params.shape))
        # params = params.view(B, 3, -1, N)
        if reduced_params:
            learn_theta = params[:, :4, :].permute(0, 2, 1).contiguous()
            # print("learn_theta.shape : " + str(learn_theta.shape))
            learn_t = params[:, 4:, :].permute(0, 2, 1).contiguous()
            # print("learn_t.shape : " + str(learn_theta.shape))
            # input()
            #learn_r, learn_t = self.recover_theta_to_extri(learn_theta, learn_t)
            learn_r, learn_t = self.recover_quaternion_to_extri(learn_theta, learn_t)
        else:
            learn_r = params.view(B, 3, -1, N)[:, :, :3].permute(0, 3, 1, 2).contiguous()
            learn_t = params.view(B, 3, -1, N)[:, :, 3:].permute(0, 3, 1, 2).contiguous()
        #learn_t[:, :-1, -1, -1]= torch.tanh(learn_t[:, :-1, -1, -1])*0.5
        idx = 0
        all_extri = []
        all_extri_inv = []
        all_proj = []
        loss_learn_t = []
        #print(subjects[0])
        exi_mat = torch.zeros(B, N, 3, 4)
        # print(learn_r[idx].shape)
        # print(learn_t[idx].shape)
        exi_mat[:, :, :, :3] = learn_r
        exi_mat[:, :, :, -1] = torch.einsum('bnjc,bnkc->bnk', -learn_t.view(B, N, 1, 3), learn_r)
        exi_mat_inv = torch.zeros(B, N, 3, 4)
        exi_mat_inv[:, :, :, :3] = learn_r.permute(0, 1,  3, 2).contiguous()  # torch.linalg.inv(learn_r[idx])
        exi_mat_inv[:, : , :, -1] = learn_t.view(B, N, 3)


        K_batch = []
        for subject in subjects:
            K_batch.append(torch.unsqueeze(self.camera_set[subject[0]]['K'], 0))
        K_batch = torch.cat(K_batch, 0)
        ######world->image
        prj_mat = torch.einsum('bnkj,bnjc->bnkc', K_batch, exi_mat)  # (N_view, 3, 4)

        #return all_extri, all_extri_inv, all_proj
        return exi_mat, exi_mat_inv, prj_mat

    def recover_extri_extri_inv(self, learn_theta, learn_t):
        learn_r, learn_t = self.recover_theta_to_extri(learn_theta, learn_t)

        idx = 0
        all_extri={}
        all_extri_inv={}
        for subject in self._cameras.keys():
            cameras = self._cameras[subject]
            N_cam = len(cameras)
            #####world->camera
            exi_mat = torch.zeros(N_cam, 3, 4)
            #print(learn_r[idx].shape)
            exi_mat[:, :, :3] = learn_r[idx]
            exi_mat[:, :, -1] = torch.einsum('njc,nkc->nk', -learn_t[idx].view(4, 1, 3), learn_r[idx])
            all_extri[subject]= exi_mat
            #####camera->world
            exi_mat_inv = torch.zeros(N_cam, 3, 4)
            exi_mat_inv[:, :, :3] = learn_r[idx].permute(0, 2, 1).contiguous()
            exi_mat_inv[:, :, -1] = learn_t[idx].view(4, 3)
            all_extri_inv[subject] = exi_mat_inv
            idx = idx+1
        return all_extri, all_extri_inv




    def p3dwd_p2dim_batch(self, p3d, subject, view_list, exi_mat=None):
        #projrct 3d camera coordination to 2d image coordination, without optical distortion
        B, J, C = p3d.shape
        prj_mat = torch.zeros(B, 4, 3, 4)
        for inx, sub in enumerate(subject):
            prj_mat[inx] = self.camera_set[sub[0]]['prj_mat']
        p_3d_homo = torch.cat((p3d, torch.ones(p3d.shape[0], 17, 1).to(p3d.device)), dim = -1)
        p_2dim_homo = torch.einsum('tnqc,tjc->tnjq',prj_mat.to(p3d.device), p_3d_homo)
        p_2dim = p_2dim_homo[...,:2] / p_2dim_homo[...,-1:]
        p_2dim = p_2dim[...,:2]
        return p_2dim

    def p3dwd_p3dcam_batch(self, p3d, subject, view_list, extri=None,is_predicted_params=False):
        B, J, C = p3d.shape

        if self.cfg.TRAIN.LEARN_CAM_PARM:
            exi_mat = []
        else:
            exi_mat = torch.zeros(B, len(view_list), 3, 4)
        K = torch.zeros(B, len(view_list), 3, 3)
        for inx, sub in enumerate(subject):
            if self.cfg.TRAIN.LEARN_CAM_PARM:
                if is_predicted_params:
                    exi_mat.append(torch.unsqueeze(extri[inx], dim=0))
                else:
                    exi_mat.append(torch.unsqueeze(extri[sub[0]], dim=0))
            else:
                exi_mat[inx] = self.camera_set[sub[0]]['exi_mat'][view_list]
            K[inx] = self.camera_set[sub[0]]['K'][view_list]
        if self.cfg.TRAIN.LEARN_CAM_PARM:
            exi_mat = torch.cat(exi_mat, dim=0)
        p_3d_homo = torch.cat((p3d, torch.ones(p3d.shape[0], 17, 1).to(p3d.device)), dim = -1)
        p_3d_homo = torch.as_tensor(p_3d_homo, dtype=torch.float32)
        exi_mat = exi_mat.to(p3d.device)
        p_3d_cam  = torch.einsum('mnkc,mjc->mnjk', exi_mat, p_3d_homo) #(T, N, J, 3)
        #if cam2im:
        #    p_2d = torch.einsum('mnjc,mnkc->mnjk', p_3d_cam, K.to(p3d.device))
        #    p_2d[:,:,:,:2] /= p_2d[:,:,:,-1:]
        #    p_2d = p_2d[...,:2]
        return p_3d_cam

class Human36mDataset(MocapDataset):
    def __init__(self, cfg, keypoints): 
        super().__init__()
        r_keypoints = copy.deepcopy(keypoints)
        keypoints = copy.deepcopy(keypoints)
        self._cameras = copy.deepcopy(h36m_cameras_extrinsic_params)
        for subject, cameras in self._cameras.items():
            for i, cam in enumerate(cameras):
                cam.update(h36m_cameras_intrinsic_params[i])
                for k, v in cam.items():
                    if k not in ['id', 'res_w', 'res_h']:
                        cam[k] = np.array(v, dtype='float32')
                
                # Normalize camera frame
                cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'], h=cam['res_h']).astype('float32')
                cam['focal_length'] = cam['focal_length']/cam['res_w']*2
                if 'translation' in cam:
                    cam['translation'] = torch.from_numpy(cam['translation'])/1000 # mm to meters
                if 'rotation' in cam:
                    cam['rotation'] = torch.FloatTensor(cam['rotation'])
                    
                # Add intrinsic parameters vector
                
                cam['project_to_2d_linear'] = torch.from_numpy(np.concatenate((cam['focal_length'],
                                                   cam['center'],
                                                   cam['radial_distortion'],
                                                   cam['tangential_distortion'])))
                
        mpjpe_loss = {}
        for subject, sub_data in keypoints.items():
            print('***', subject)
            mpjpe_loss[subject] = {}
            cameras = self._cameras[subject]
            for action, act_data in sub_data.items():
                temp_data = None
                temp_cam = None
                temp_r = None
                temp_t = None
                for cam, cam_data in enumerate(act_data):
                    camera = cameras[cam]
                    T, J, C = cam_data.shape#(p2d_gt, p2d_pre, p3d)
                    in_param = camera['project_to_2d_linear'].view(1, -1)
                    cam_data = cam_data.view(T, 1, J, C)
                    r = camera['rotation'].view(1, 3, 3)
                    t = camera['translation'].view(1, 3)
                    
                    if temp_data is None:
                        temp_data = cam_data
                        temp_cam = in_param
                        temp_r = r
                        temp_t = t
                    else:
                        temp_data = torch.cat((temp_data, cam_data), dim = 1)
                        temp_cam = torch.cat((temp_cam, in_param), dim = 0)
                        temp_r = torch.cat((temp_r, r), dim = 0)
                        temp_t = torch.cat((temp_t, t), dim = 0)
                        
                p3d = temp_data[:,:,:,4:7]
                p3d[:,:,1:] += p3d[:,:,:1]
                pred_2d = temp_data[:,:,:,2:4]
                gt_2d = temp_data[:,:,:,0:2]
                N_cam = temp_cam.shape[0]
                temp_cam = temp_cam.view(1, N_cam, 1, -1)
                
                #####world->camera
                exi_mat = torch.zeros(N_cam, 3, 4)
                exi_mat[:,:,:3] = temp_r
                exi_mat[:,:,-1] = torch.einsum('njc,nkc->nk',  -temp_t.view(4, 1, 3), temp_r) 
                #####camera->world
                exi_mat_inv = torch.zeros(N_cam, 3, 4)
                exi_mat_inv[:,:,:3] = temp_r.permute(0, 2, 1).contiguous()
                exi_mat_inv[:,:,-1] = temp_t.view(4,3)
                #####camera->image
                K = torch.zeros(N_cam, 3, 3)
                K[:, 0, 0] = temp_cam[0,:,0,0]
                K[:,1,1] = temp_cam[0,:,0,1]
                K[:,0,2] = temp_cam[0,:,0,2]
                K[:,1,2] = temp_cam[0,:,0,3]
                K[:,2,2] = 1
                ######world->image
                prj_mat = torch.einsum('nkj,njc->nkc',K, exi_mat)
                if cfg.DATA.USE_GT_2D:
                    temp_pred_2d = gt_2d.permute(0, 2, 1, 3) #(T, J, N, C)
                else:
                    temp_pred_2d = pred_2d.permute(0, 2, 1, 3) #(T, J, N, C)
                temp_pred_2d = temp_pred_2d.contiguous().view(-1, N_cam, 2) #(T*J, N, 2)
                for num_view in list(range(2, N_cam+1)):
                    for view_list in itertools.combinations(list(range(N_cam)), num_view):
                        trj_w3d = triangulation(temp_pred_2d[:,view_list].view(-1, 17, len(view_list), 2), prj_mat[view_list,...])

                        trj_w3d_homo = torch.cat((trj_w3d, torch.ones(trj_w3d.shape[0], 17, 1)), dim = -1)
                        trj_c3d = torch.einsum('nkc,mjc->mnjk', exi_mat[view_list, ...], trj_w3d_homo) #(T, N, J, 3)
                        
                        trj_2d = torch.einsum('mnjc,nkc->mnjk', trj_c3d, K[view_list, ...])
                        trj_2d[:,:,:,:2] /= trj_2d[:,:,:,-1:]
                        trj_2d = trj_2d[...,:2]
                        pred = trj_c3d - trj_c3d[:,:,:1] #(T, N, J, 3)
                        target = p3d[:,view_list] - p3d[:,view_list,:1]
                        pred = pred.contiguous().view(-1, 17, 3)
                        target = target.contiguous().view(-1, 17, 3)
                        if cfg.TEST.METRIC == 'mpjpe':
                            loss = mpjpe(pred, target)
                        else:
                            loss = p_mpjpe(cfg, pred, target)
                        trj_c3d[:,:,1:] -= trj_c3d[:,:,:1]
                        if action not in mpjpe_loss[subject]:
                            mpjpe_loss[subject][action] = {'T_length':trj_c3d.shape[0], 'N_view':{}}
                        
                        if len(view_list) not in mpjpe_loss[subject][action]['N_view']:
                            mpjpe_loss[subject][action]['N_view'][len(view_list)] = {'avg':0, 'N':0}
                        found_n_view_loss = mpjpe_loss[subject][action]['N_view'][len(view_list)]
                        
                        for view_id in view_list:
                            if view_id not in found_n_view_loss:
                                found_n_view_loss[view_id] = {'mpjpe':loss.item(), 'N':1}
                            else:
                                found_n_view_loss[view_id]['mpjpe'] += loss.item()
                                found_n_view_loss[view_id]['N'] += 1
                #break            
        for sub, sub_loss in mpjpe_loss.items():
            for act, act_loss in sub_loss.items():
                for n_view, n_view_loss in act_loss['N_view'].items():
                    for view_id, view_loss in n_view_loss.items():
                        if view_id == 'N' or view_id == 'avg':
                            continue
                        
                        view_loss['avg'] = view_loss['mpjpe'] / view_loss['N']
                        n_view_loss['avg'] += view_loss['mpjpe']
                        n_view_loss['N'] += view_loss['N']
                    n_view_loss['avg'] /= n_view_loss['N']
           
                        

        self.data = r_keypoints
        self.mpjpe_loss = mpjpe_loss
        
class Human36mCamDataset(MocapDataset):
    def __init__(self, keypoints): 
        super().__init__()
        self.r_keypoints = copy.deepcopy(keypoints)
        keypoints = copy.deepcopy(keypoints)
        self._cameras = copy.deepcopy(h36m_cameras_extrinsic_params)
        for subject, cameras in self._cameras.items():
            for i, cam in enumerate(cameras):
                cam.update(h36m_cameras_intrinsic_params[i])
                for k, v in cam.items():
                    if k not in ['id', 'res_w', 'res_h']:
                        cam[k] = np.array(v, dtype='float32')
                
                # Normalize camera frame
                cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'], h=cam['res_h']).astype('float32')
                cam['focal_length'] = cam['focal_length']/cam['res_w']*2
                if 'translation' in cam:
                    cam['translation'] = torch.from_numpy(cam['translation'])/1000 # mm to meters
                if 'rotation' in cam:
                    cam['rotation'] = torch.FloatTensor(cam['rotation'])
                    
                # Add intrinsic parameters vector
                
                cam['project_to_2d_linear'] = torch.from_numpy(np.concatenate((cam['focal_length'],
                                                   cam['center'],
                                                   cam['radial_distortion'],
                                                   cam['tangential_distortion'])))
        #####norm
        max_norm = 0
        for subject, sub_data in keypoints.items():
            if subject not in ['S1', 'S5', 'S6', 'S7', 'S8']:
                continue
            print(subject)
            for action, act_data in sub_data.items():
                for cam, cam_data in enumerate(act_data):
                    p3d = copy.deepcopy(cam_data[:,:,4:7]) #(T, J, 7)
                    p3d[:,1:] += p3d[:,:1]
                    norm_3d_gt = p3d / p3d[:,:,-1:] #(T, J, 3)
                    max_x = torch.max(torch.abs(norm_3d_gt[:,:,0]))
                    max_y = torch.max(torch.abs(norm_3d_gt[:,:,1]))
                    max_norm = max(max_norm, max_x, max_y)
        self.max_norm = max_norm         
        self.r_camera_param = {}        
        for subject, sub_data in keypoints.items():
            print('***', subject)
            cameras = self._cameras[subject]
            N_cam = len(cameras)
            temp_cam = None
            temp_r = None
            temp_t = None
            for cam in range(N_cam):
                camera = cameras[cam]
                in_param = camera['project_to_2d_linear'].view(1, -1)
                r = camera['rotation'].view(1, 3, 3)
                t = camera['translation'].view(1, 3)
                if temp_cam is None:
                    temp_cam = in_param
                    temp_r = r
                    temp_t = t
                else:
                    temp_cam = torch.cat((temp_cam, in_param), dim = 0)
                    temp_r = torch.cat((temp_r, r), dim = 0)
                    temp_t = torch.cat((temp_t, t), dim = 0)
            temp_cam = temp_cam.view(1, N_cam, 1, -1)
            #####world->camera
            exi_mat = torch.zeros(N_cam, 3, 4)
            exi_mat[:,:,:3] = temp_r
            exi_mat[:,:,-1] = torch.einsum('njc,nkc->nk',  -temp_t.view(4, 1, 3), temp_r) 
            
            #####camera->world
            exi_mat_inv = torch.zeros(N_cam, 3, 4)
            exi_mat_inv[:,:,:3] = temp_r.permute(0, 2, 1).contiguous()
            exi_mat_inv[:,:,-1] = temp_t.view(4,3)
            #####camera->image
            K = torch.zeros(N_cam, 3, 3)
            K[:, 0, 0] = temp_cam[0,:,0,0] #f_x = f * dx
            K[:,1,1] = temp_cam[0,:,0,1]  #f_y = f * dy
            K[:,0,2] = temp_cam[0,:,0,2]  #cx
            K[:,1,2] = temp_cam[0,:,0,3]  #cy
            K[:,2,2] = 1
            ######world->image
            prj_mat = torch.einsum('nkj,njc->nkc',K, exi_mat)
            
            ######image->norm_3d  (x_cam / z_cam, y_cam / z_cam, 1)
            K_inv = torch.zeros(N_cam, 3, 3)
            K_inv[:, 0, 0] = 1.0 / temp_cam[0,:,0,0] #1 / f_x
            K_inv[:,1,1] = 1.0 / temp_cam[0,:,0,1]  #1 / f_y
            K_inv[:,0,2] = -temp_cam[0,:,0,2] / temp_cam[0,:,0,0]  #- cx / f_x
            K_inv[:,1,2] = -temp_cam[0,:,0,3] / temp_cam[0,:,0,1]  #- cy / f_y
            K_inv[:,2,2] = 1
            
            self.r_camera_param[subject] = {'world_cam':exi_mat, 'cam_world':exi_mat_inv}
            for action, act_data in sub_data.items():
                temp_data = None
                for cam, cam_data in enumerate(act_data):
                    T, J, C = cam_data.shape#(p2d_gt, p2d_pre, p3d_gt)
                    cam_data = cam_data.view(T, 1, J, C)
                    
                    if temp_data is None:
                        temp_data = cam_data
                    else:
                        temp_data = torch.cat((temp_data, cam_data), dim = 1)
                        
                p3d = temp_data[:,:,:,4:7]
                p3d[:,:,1:] += p3d[:,:,:1]
                pred_2d = temp_data[:,:,:,2:4]
                gt_2d = temp_data[:,:,:,0:2]
                
                
                #pred_2d: (T, N, J, 2)  p3d；(T, N, J, 3)
                temp_pred_2d = pred_2d.permute(0, 2, 1, 3).contiguous() #(T, J, N, C)
                temp_pred_2d = temp_pred_2d.contiguous().view(-1, N_cam, 2) #(T*J, N, 2)
                pred_2d_homo = torch.cat((pred_2d, torch.ones(pred_2d.shape[0], N_cam, 17, 1)), dim = -1)
                cam_norm_3d_trj = torch.einsum('tnjc,nqc->tnjq', pred_2d_homo, K_inv)
                norm_3d_gt = p3d / p3d[:,:,:,-1:]
                
                cam_norm_3d_trj_homo = torch.cat((cam_norm_3d_trj, torch.ones(cam_norm_3d_trj.shape[0], N_cam, 17, 1)), dim = -1)
                #torch.ones(cam_norm_3d_trj.shape[0], N_cam, 17, 1).shape: torch.Size([2478, 4, 17, 1])
                #cam_norm_3d_trj.shape: torch.Size([2478, 4, 17, 3])
                p3d_homo = torch.cat((p3d, torch.ones(p3d.shape[0], N_cam, 17, 1)), dim = -1)
                #p3d.shape: torch.Size([2478, 4, 17, 3])
                #p3d_homo.shape: torch.Size([2478, 4, 17, 4])

                world_norm_3d_trj = torch.einsum('nqc,tnjc->tnjq', exi_mat_inv, cam_norm_3d_trj_homo)
                #exi_mat_inv.shape: torch.Size([4, 3, 4])
                #cam_norm_3d_trj_homo.shape: torch.Size([2478, 4, 17, 4])
                world_3d = torch.einsum('nqc,tnjc->tnjq', exi_mat_inv, p3d_homo)
                #world_norm_3d_trj.shape:torch.Size([2478, 4, 17, 3])
                for cam in range(N_cam):
                    self.r_keypoints[subject][action][cam][:,:,[2, 3]] = cam_norm_3d_trj[:,cam,:,:2]
                
    def get_data(self):
        return self.r_keypoints
    def get_camera(self):
        return self.r_camera_param
    def get_norm(self):
        return self.max_norm
                    
