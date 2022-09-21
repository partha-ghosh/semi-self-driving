from run_utils import *

root = os.path.dirname(os.path.abspath(__file__))
exp_time = '09_08_14_15_40' #datetime.datetime.now().strftime("%m_%d_%H_%M_%S")

def gen_config(
    training_type, #s,ss,ssf,ssgt
    eval, 
    dataloader=1, # 0 -> old, 1 -> new
    dataset=0, # 0 -> minimal, 1 -> transfuser+ 
    use_acc=0,
    use_nav=0,
    imgaug=1,
    what_if=0,
    use_target_point=1,
    predict_confidence=0,
    confidence_threshold=0.33,
    supervised_towns=[1,2,3,4,6,7,10],
    self_supervised_towns=[1],
    script_dir=f'{root}/ssd',
    epochs=50,
    batch_size=64,
    agent='aim_agent',
    copy_last_model=0,
    load_model=0, # or path/to/pth, 1 -> load from test_dir
    logdir='log',
    device='cuda',
    val_every=5,
    lr=2e-5,
    **kwargs, # for test_name=test
    ):

    if kwargs.get('test_name', None):
        test_name = kwargs.get('test_name')
    else:
        test_name = f'd{dataset}_dl{dataloader}_imgaug{imgaug}_nav{use_nav}_tp{use_target_point}_pc{predict_confidence}_thresh{confidence_threshold}_whatif{what_if}_{training_type}_st{len(supervised_towns)}_sst{len(self_supervised_towns)}'
    
    root_data_dir = '/mnt/qb/work/geiger/pghosh58/transfuser/data'
    data_dir = f'{root_data_dir}/filtered_14_weathers_minimal_data' if dataset==0 else f'{root_data_dir}/filtered_transfuser_plus_data'

    train_towns = [f'Town{str(i).zfill(2)}' for i in supervised_towns]
    ssd_towns = [f'Town{str(i).zfill(2)}' for i in self_supervised_towns]
    val_towns = ['Town05']

    train_data, val_data, ssd_data = [], [], []
    for town in train_towns:
        # train_data.append(town+'_tiny')
        # train_data.append(town+'_short')
        # train_data.append(town+'_long')
        train_data.append(town)
    
    for town in ssd_towns:
        # ssd_data.append(town+'_tiny')
        # ssd_data.append(town+'_short')
        # ssd_data.append(town+'_long')
        ssd_data.append(town)

    for town in val_towns:
        # val_data.append(town+'_short')
        val_data.append(town)
        

    config = dict(
        test_id=None,
        test_name=test_name,
        test_dir=f'{root}/tmp/{test_name}',
        script_dir=script_dir,
        data_dir=data_dir,
        supervised_towns=train_data,
        self_supervised_towns=ssd_data,
        validation_towns=val_data,
        training_type=training_type,
        dataloader=dataloader,
        imgaug=imgaug,
        use_acc=use_acc,
        use_nav=use_nav,
        use_target_point=use_target_point,
        predict_confidence=predict_confidence,
        confidence_threshold=confidence_threshold,
        what_if=what_if,
        eval=eval,
        agent_name=agent,
        epochs=epochs,
        batch_size=batch_size,
        copy_last_model=copy_last_model,
        load_model=load_model,
        logdir=logdir,
        device=device,
        val_every=val_every,
        lr=lr,

        seq_len = 1, # input timesteps
        pred_len = 4, # future waypoints predicted
        
        ignore_sides = True, # don't consider side cameras
        ignore_rear = True, # don't consider rear cameras

        input_resolution = 256,

        scale = 1, # image pre-processing
        crop = 256, # image pre-processing

        # Controller
        turn_KP = 1.25,
        turn_KI = 0.75,
        turn_KD = 0.3,
        turn_n = 40, # buffer size

        speed_KP = 5.0,
        speed_KI = 0.5,
        speed_KD = 1.0,
        speed_n = 40, # buffer size

        max_throttle = 0.75, # upper limit on throttle signal value in dataset
        brake_speed = 0.4, # desired speed below which brake is triggered
        brake_ratio = 1.1, # ratio of speed to desired speed at which brake is triggered
        clip_delta = 0.25, # maximum change in speed input to logitudinal controller
    )

    return config



tests = [
    [
        # gen_config(training_type='s',eval=2,epochs=2,test_name='test',val_every=1,)
        # gen_config(training_type='s',use_acc=2, eval=0,epochs=0,test_name='test',val_every=1,)
        # gen_config(training_type='s',eval=3,epochs=15, load_model=True)
        # gen_config(training_type='s',eval=3,use_acc=2,)
    ],

    # [gen_config(training_type='s',dataset=0,dataloader=0,imgaug=0,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,eval=3,epochs=50,val_every=5,load_model=0)],
    # [gen_config(training_type='s',dataset=0,dataloader=0,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,eval=3,epochs=50,val_every=5,load_model=0)],
    # [gen_config(training_type='s',dataset=0,dataloader=1,imgaug=0,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,eval=3,epochs=50,val_every=5,load_model=0)],
    # [gen_config(training_type='s',dataset=1,dataloader=0,imgaug=0,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,eval=3,epochs=50,val_every=5,load_model=0)],
#    [gen_config(training_type='s',dataset=0,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,eval=3,epochs=50,val_every=5,load_model=0)],
   [gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,eval=3,epochs=50,val_every=5,load_model=0)],
    # [gen_config(training_type='s',dataset=1,dataloader=0,imgaug=0,use_nav=1,use_target_point=0,predict_confidence=0,confidence_threshold=0.33,eval=3,epochs=50,val_every=5,load_model=0)],
    # [gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=1,use_target_point=0,predict_confidence=0,confidence_threshold=0.33,eval=3,epochs=50,val_every=5,load_model=0)],
#    [gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,eval=3,epochs=50,val_every=5,load_model=0,supervised_towns=[1,2,3,4])],
#    [gen_config(training_type='s',dataset=0,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,eval=3,epochs=50,val_every=5,load_model=0,supervised_towns=[1,2,3,4])],
    # [gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=1,confidence_threshold=0.33,eval=3,epochs=50,val_every=5,load_model=0,supervised_towns=[1,2,3,4])],
    # [gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=1,use_target_point=0,predict_confidence=0,confidence_threshold=0.33,eval=3,epochs=50,val_every=5,load_model=0,supervised_towns=[1,2,3,4])],
    # [gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=1,use_target_point=0,predict_confidence=1,confidence_threshold=0.33,eval=3,epochs=50,val_every=5,load_model=0,supervised_towns=[1,2,3,4])],



    # [gen_config(training_type='s',dataset=0,dataloader=0,imgaug=0,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,eval=3,epochs=0,val_every=5,load_model=1)],
    # [gen_config(training_type='s',dataset=0,dataloader=0,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,eval=3,epochs=0,val_every=5,load_model=1)],
    # [gen_config(training_type='s',dataset=0,dataloader=1,imgaug=0,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,eval=3,epochs=0,val_every=5,load_model=1)],
    # [gen_config(training_type='s',dataset=1,dataloader=0,imgaug=0,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,eval=3,epochs=0,val_every=5,load_model=1)],
    # [gen_config(training_type='s',dataset=0,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,eval=3,epochs=0,val_every=5,load_model=1)],
    # [gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,eval=3,epochs=0,val_every=5,load_model=1)],
    # [gen_config(training_type='s',dataset=1,dataloader=0,imgaug=0,use_nav=1,use_target_point=0,predict_confidence=0,confidence_threshold=0.33,eval=3,epochs=0,val_every=5,load_model=1)],
    # [gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=1,use_target_point=0,predict_confidence=0,confidence_threshold=0.33,eval=3,epochs=0,val_every=5,load_model=1)],
    # [gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,eval=3,epochs=0,val_every=5,load_model=1,supervised_towns=[1,2,3,4])],
    # [gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=1,confidence_threshold=0.33,eval=3,epochs=0,val_every=5,load_model=1,supervised_towns=[1,2,3,4])],
    # [gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=1,use_target_point=0,predict_confidence=0,confidence_threshold=0.33,eval=3,epochs=0,val_every=5,load_model=1,supervised_towns=[1,2,3,4])],
    # [gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=1,use_target_point=0,predict_confidence=1,confidence_threshold=0.33,eval=3,epochs=0,val_every=5,load_model=1,supervised_towns=[1,2,3,4])],


    # [
    #     gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,what_if=0,eval=0,epochs=0,val_every=1,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],load_model=f'{root}/tmp/d1_dl1_imgaug1_nav0_tp1_pc0_thresh0.33_whatif0_s_st4_sst1/log/saved_model/model.pth'),
    #     gen_config(training_type='ssf',dataset=1,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,what_if=0,eval=3,epochs=100,val_every=5,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],copy_last_model=1,load_model=1),
    # ],

    # [
    #     gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,what_if=3,eval=0,epochs=0,val_every=1,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],load_model=f'{root}/tmp/d1_dl1_imgaug1_nav0_tp1_pc0_thresh0.33_whatif0_s_st4_sst1/log/saved_model/model.pth'),
    #     gen_config(training_type='ssf',dataset=1,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,what_if=3,eval=3,epochs=100,val_every=5,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],copy_last_model=1,load_model=1),
    # ],

    # [
    #     gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=1,confidence_threshold=0.33,what_if=0,eval=0,epochs=0,val_every=1,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],load_model=f'{root}/tmp/d1_dl1_imgaug1_nav0_tp1_pc1_thresh0.33_whatif0_s_st4_sst1/log/saved_model/model.pth'),
    #     gen_config(training_type='ssf',dataset=1,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=1,confidence_threshold=0.33,what_if=0,eval=3,epochs=100,val_every=5,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],copy_last_model=1,load_model=1),
    # ],

    # [
    #     gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=1,confidence_threshold=0.33,what_if=3,eval=0,epochs=0,val_every=1,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],load_model=f'{root}/tmp/d1_dl1_imgaug1_nav0_tp1_pc1_thresh0.33_whatif0_s_st4_sst1/log/saved_model/model.pth'),
    #     gen_config(training_type='ssf',dataset=1,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=1,confidence_threshold=0.33,what_if=3,eval=3,epochs=100,val_every=5,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],copy_last_model=1,load_model=1),
    # ],




    # [
    #     gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=1,use_target_point=0,predict_confidence=0,confidence_threshold=0.33,what_if=0,eval=0,epochs=0,val_every=1,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],load_model=f'{root}/tmp/d1_dl1_imgaug1_nav1_tp0_pc0_thresh0.33_whatif0_s_st4_sst1/log/saved_model/model.pth'),
    #     gen_config(training_type='ssf',dataset=1,dataloader=1,imgaug=1,use_nav=1,use_target_point=0,predict_confidence=0,confidence_threshold=0.33,what_if=0,eval=3,epochs=100,val_every=5,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],copy_last_model=1,load_model=1),
    # ],

    # [
    #     gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=1,use_target_point=0,predict_confidence=0,confidence_threshold=0.33,what_if=3,eval=0,epochs=0,val_every=1,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],load_model=f'{root}/tmp/d1_dl1_imgaug1_nav1_tp0_pc0_thresh0.33_whatif0_s_st4_sst1/log/saved_model/model.pth'),
    #     gen_config(training_type='ssf',dataset=1,dataloader=1,imgaug=1,use_nav=1,use_target_point=0,predict_confidence=0,confidence_threshold=0.33,what_if=3,eval=3,epochs=100,val_every=5,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],copy_last_model=1,load_model=1),
    # ],

    # [
    #     gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=1,use_target_point=0,predict_confidence=1,confidence_threshold=0.33,what_if=0,eval=0,epochs=0,val_every=1,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],load_model=f'{root}/tmp/d1_dl1_imgaug1_nav1_tp0_pc1_thresh0.33_whatif0_s_st4_sst1/log/saved_model/model.pth'),
    #     gen_config(training_type='ssf',dataset=1,dataloader=1,imgaug=1,use_nav=1,use_target_point=0,predict_confidence=1,confidence_threshold=0.33,what_if=0,eval=3,epochs=100,val_every=5,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],copy_last_model=1,load_model=1),
    # ],

    # [
    #     gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=1,use_target_point=0,predict_confidence=1,confidence_threshold=0.33,what_if=3,eval=0,epochs=0,val_every=1,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],load_model=f'{root}/tmp/d1_dl1_imgaug1_nav1_tp0_pc1_thresh0.33_whatif0_s_st4_sst1/log/saved_model/model.pth'),
    #     gen_config(training_type='ssf',dataset=1,dataloader=1,imgaug=1,use_nav=1,use_target_point=0,predict_confidence=1,confidence_threshold=0.33,what_if=3,eval=3,epochs=100,val_every=5,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],copy_last_model=1,load_model=1),
    # ],



    # [gen_config(training_type='s',dataset=0,dataloader=0,imgaug=0,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,eval=0,epochs=2,val_every=1)],
    # [gen_config(training_type='s',dataset=0,dataloader=0,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,eval=0,epochs=2,val_every=1)],
    # [gen_config(training_type='s',dataset=0,dataloader=1,imgaug=0,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,eval=0,epochs=2,val_every=1)],
    # [gen_config(training_type='s',dataset=1,dataloader=0,imgaug=0,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,eval=0,epochs=2,val_every=1)],
    
    # [gen_config(training_type='s',dataset=0,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,eval=0,epochs=2,val_every=1)],
    # [gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,eval=0,epochs=2,val_every=1)],
    
    # [gen_config(training_type='s',dataset=1,dataloader=0,imgaug=0,use_nav=1,use_target_point=0,predict_confidence=0,confidence_threshold=0.33,eval=0,epochs=2,val_every=1)],
    # [gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=1,use_target_point=0,predict_confidence=0,confidence_threshold=0.33,eval=0,epochs=2,val_every=1)],
    
    # [gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,eval=0,epochs=2,val_every=1,supervised_towns=[1,2,3,4])],
    # [gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=1,confidence_threshold=0.33,eval=0,epochs=2,val_every=1,supervised_towns=[1,2,3,4])],

    # [gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=1,use_target_point=0,predict_confidence=0,confidence_threshold=0.33,eval=0,epochs=2,val_every=1,supervised_towns=[1,2,3,4])],
    # [gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=1,use_target_point=0,predict_confidence=1,confidence_threshold=0.33,eval=0,epochs=2,val_every=1,supervised_towns=[1,2,3,4])],


    # [
    #     gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,what_if=0,eval=0,epochs=0,val_every=1,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],load_model=f'{root}/tmp/d1_dl1_imgaug1_nav0_tp1_pc0_thresh0.33_whatif0_s_st4_sst1/log/saved_model/model.pth'),
    #     gen_config(training_type='ssf',dataset=1,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,what_if=0,eval=0,epochs=2,val_every=1,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],copy_last_model=1,load_model=1),
    # ],

    # [
    #     gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,what_if=3,eval=0,epochs=0,val_every=1,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],load_model=f'{root}/tmp/d1_dl1_imgaug1_nav0_tp1_pc0_thresh0.33_whatif0_s_st4_sst1/log/saved_model/model.pth'),
    #     gen_config(training_type='ssf',dataset=1,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=0,confidence_threshold=0.33,what_if=3,eval=0,epochs=2,val_every=1,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],copy_last_model=1,load_model=1),
    # ],

    # [
    #     gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=1,confidence_threshold=0.33,what_if=0,eval=0,epochs=0,val_every=1,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],load_model=f'{root}/tmp/d1_dl1_imgaug1_nav0_tp1_pc1_thresh0.33_whatif0_s_st4_sst1/log/saved_model/model.pth'),
    #     gen_config(training_type='ssf',dataset=1,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=1,confidence_threshold=0.33,what_if=0,eval=0,epochs=2,val_every=1,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],copy_last_model=1,load_model=1),
    # ],

    # [
    #     gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=1,confidence_threshold=0.33,what_if=3,eval=0,epochs=0,val_every=1,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],load_model=f'{root}/tmp/d1_dl1_imgaug1_nav0_tp1_pc1_thresh0.33_whatif0_s_st4_sst1/log/saved_model/model.pth'),
    #     gen_config(training_type='ssf',dataset=1,dataloader=1,imgaug=1,use_nav=0,use_target_point=1,predict_confidence=1,confidence_threshold=0.33,what_if=3,eval=0,epochs=2,val_every=1,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],copy_last_model=1,load_model=1),
    # ],




    # [
    #     gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=1,use_target_point=0,predict_confidence=0,confidence_threshold=0.33,what_if=0,eval=0,epochs=0,val_every=1,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],load_model=f'{root}/tmp/d1_dl1_imgaug1_nav1_tp0_pc0_thresh0.33_whatif0_s_st4_sst1/log/saved_model/model.pth'),
    #     gen_config(training_type='ssf',dataset=1,dataloader=1,imgaug=1,use_nav=1,use_target_point=0,predict_confidence=0,confidence_threshold=0.33,what_if=0,eval=0,epochs=2,val_every=1,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],copy_last_model=1,load_model=1),
    # ],

    # [
    #     gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=1,use_target_point=0,predict_confidence=0,confidence_threshold=0.33,what_if=3,eval=0,epochs=0,val_every=1,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],load_model=f'{root}/tmp/d1_dl1_imgaug1_nav1_tp0_pc0_thresh0.33_whatif0_s_st4_sst1/log/saved_model/model.pth'),
    #     gen_config(training_type='ssf',dataset=1,dataloader=1,imgaug=1,use_nav=1,use_target_point=0,predict_confidence=0,confidence_threshold=0.33,what_if=3,eval=0,epochs=2,val_every=1,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],copy_last_model=1,load_model=1),
    # ],

    # [
    #     gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=1,use_target_point=0,predict_confidence=1,confidence_threshold=0.33,what_if=0,eval=0,epochs=0,val_every=1,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],load_model=f'{root}/tmp/d1_dl1_imgaug1_nav1_tp0_pc1_thresh0.33_whatif0_s_st4_sst1/log/saved_model/model.pth'),
    #     gen_config(training_type='ssf',dataset=1,dataloader=1,imgaug=1,use_nav=1,use_target_point=0,predict_confidence=1,confidence_threshold=0.33,what_if=0,eval=0,epochs=2,val_every=1,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],copy_last_model=1,load_model=1),
    # ],

    # [
    #     gen_config(training_type='s',dataset=1,dataloader=1,imgaug=1,use_nav=1,use_target_point=0,predict_confidence=1,confidence_threshold=0.33,what_if=3,eval=0,epochs=0,val_every=1,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],load_model=f'{root}/tmp/d1_dl1_imgaug1_nav1_tp0_pc1_thresh0.33_whatif0_s_st4_sst1/log/saved_model/model.pth'),
    #     gen_config(training_type='ssf',dataset=1,dataloader=1,imgaug=1,use_nav=1,use_target_point=0,predict_confidence=1,confidence_threshold=0.33,what_if=3,eval=0,epochs=2,val_every=1,supervised_towns=[1,2,3,4],self_supervised_towns=[1,2,3,4,6,7,10],copy_last_model=1,load_model=1),
    # ],

]

# os.system('pkill -f carla')
# time.sleep(5)

def run_test(tests):
    for test in tests:
        test_name = test['test_name']
        test_dir = test['test_dir']
        script_dir = test['script_dir']
        os.system(f'mkdir -p {test_dir} && cd {script_dir} && cp trainer.py data.py model.py train.py utils.py {test_dir}/')

    test_id = id(tests)
    cmd_trains = []
    cmd_evals = []
    old_test_dir = None
    for i, test in enumerate(tests):
        # test_name = "07_30_10_59-aim-baseline-supervised_e60_b64"
        test_name = test['test_name']
        test_dir = test['test_dir']
        script_dir = test['script_dir']

        os.system(f'mkdir -p {test["data_dir"]}/pseudo/{test_id}')
        test['pseudo_data'] = f'{test["data_dir"]}/pseudo/{test_id}/processed_data.npy'
        test['test_id'] = test_id

        cmd_trains.append({
            'test_name':test_name,
            'script_dir':script_dir,
            'test_dir':test_dir,
            'cmd': 
            [
                # f'mkdir -p {test_dir} && rsync -a {script_dir}/* {test_dir}/ --exclude=log* --exclude=__pycache__',
                f'rsync -av {old_test_dir}/log {test_dir}/ --exclude=*.err --exclude=*.out --exclude=*tfevents*' if test['copy_last_model'] else "",
                f'cd {test_dir}',
                f'CUDA_VISIBLE_DEVICES=0 python train.py "{str(test)}"',
                f'''{f"cp -r {test['pseudo_data']} {test['logdir']}/" if test["training_type"]=="s" else "echo"}''',

                # 3 evaluations
                *([f'python {root}/tools/sbatch_submitter.py "sbatch {root}/shell_scripts/run_eval_{test_name}.sh"',]*test["eval"]),
                
                f'python {root}/tools/sbatch_submitter.py "sbatch {root}/shell_scripts/run_train_{tests[i+1]["test_name"]}.sh"' if i<len(tests)-1 else "",

                f'mv {root}/tmp/$SLURM_JOB_ID.out {test_dir}/log/',
                f'mv {root}/tmp/$SLURM_JOB_ID.err {test_dir}/log/',
            ]
        })
        old_test_dir = test_dir
        
        cmd_evals.append(
            {
                'test_name':test_name,
                'script_dir':script_dir,
                'test_dir':test_dir,
                'cmd':[
                    f'carla_port=`python /mnt/qb/work/geiger/pghosh58/transfuser/tools/get_carla_port.py`',
                    f'tm_port=$((port+8000))',
                    f'echo "carla port: $carla_port"',
                    f'SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 {root}/carla/CarlaUE4.sh --world-port=$carla_port -opengl &',
                    f'sleep 60',
                    f'cd {root}',
                    common_exports.format(root, test['agent_name'], test_name+'_$SLURM_JOB_ID', f'{test_dir}/log/saved_model'),
                    '{}'.format(leaderboard_evaluator.format(f'"{str(test)}"').replace("\n", " ")),# f"{test['dir']}/log/{test_name}/eval.txt"),
                    f'sleep 3',
                    f'mkdir -p {test_dir}/log/results_$SLURM_JOB_ID',
                    f'mv {root}/results/{test_name}_$SLURM_JOB_ID.json {test_dir}/log/results_$SLURM_JOB_ID/result.json',
                    f'python {root}/tools/result_parser.py --xml {root}/leaderboard/data/evaluation_routes/routes_town05_long.xml --town_maps {root}/leaderboard/data/town_maps_xodr --results {test_dir}/log/results_$SLURM_JOB_ID --save_dir {test_dir}/log/results_$SLURM_JOB_ID',
                    f'pkill -f "port=$carla_port"',

                    f'mv {root}/tmp/$SLURM_JOB_ID.out {test_dir}/log/results_$SLURM_JOB_ID/',
                    f'mv {root}/tmp/$SLURM_JOB_ID.err {test_dir}/log/results_$SLURM_JOB_ID/',
                    f'python {root}/tools/run_again.py "{test_dir}/log/results_$SLURM_JOB_ID/$SLURM_JOB_ID.err" "sbatch /mnt/qb/work/geiger/pghosh58/transfuser/shell_scripts/run_eval_{test_name}.sh"',

                ]
            }
        )

    for i, cmd_train in enumerate(cmd_trains):
        with open(f'shell_scripts/run_train_{cmd_train["test_name"]}.sh', 'w') as f:
            f.write(slurm.format(1, f'{root}/tmp',f'{root}/tmp')+"\n".join(cmd_train["cmd"]))
        os.system(f'chmod +x shell_scripts/run_train_{cmd_train["test_name"]}.sh')
        if i==0:
            os.system(f'python {root}/tools/sbatch_submitter.py "sbatch shell_scripts/run_train_{cmd_train["test_name"]}.sh"')

    for cmd_eval in cmd_evals:
        with open(f'shell_scripts/run_eval_{cmd_eval["test_name"]}.sh', 'w') as f:
            f.write(slurm.format(1, f'{root}/tmp',f'{root}/tmp')+"\n".join(cmd_eval["cmd"]))
        os.system(f'chmod +x shell_scripts/run_eval_{cmd_eval["test_name"]}.sh')

for test in tests:
    run_test(test)
