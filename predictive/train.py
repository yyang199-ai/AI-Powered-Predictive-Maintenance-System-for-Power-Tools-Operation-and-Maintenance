"""Train ablations on device-disjoint simulations and write reproducible evidence."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import TensorDataset, DataLoader
from .data import simulate_fleet, grouped_split
from .preprocessing import ConditionBaseline
from .pipeline import build_sequences
from .model import MaintenanceModel, multitask_loss


def train_one(train, validation, baseline, kind, corrected, quality_gate, epochs, seed):
    torch.manual_seed(seed)
    model = MaintenanceModel(kind, quality_gate=quality_gate)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    x, y, _ = build_sequences(train, baseline, corrected, stride=2)
    vx, vy, _ = build_sequences(validation, baseline, corrected, stride=2)
    loader = DataLoader(TensorDataset(*x, *y), batch_size=128, shuffle=True)
    best, best_loss, history = None, float('inf'), []
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            loss = multitask_loss(model(*batch[:3]), *batch[3:])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            score = float(multitask_loss(model(*vx), *vy))
        history.append(score)
        if score < best_loss:
            best_loss, best = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    model.eval()
    return model, history


def evaluate(model, data, baseline, corrected):
    x, y, indices = build_sequences(data, baseline, corrected)
    with torch.inference_mode():
        logits, hi, rul = model(*x)
    predicted = logits.argmax(dim=1).numpy()
    truth = y[0].numpy()
    actual_rul, predicted_rul = y[2].numpy(), rul[:,1].numpy()
    precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=[0,1,2,3], zero_division=0)
    error = predicted_rul - actual_rul
    # Relative error excludes near-zero lifetimes where percentages diverge.
    eligible = actual_rul >= 0.25
    samples = []
    with torch.inference_mode():
        for _ in range(10): model(*(a[:1] for a in x))
        for _ in range(100):
            start = time.perf_counter(); model(*(a[:1] for a in x)); samples.append((time.perf_counter()-start)*1000)
    return {
        'samples':len(indices), 'accuracy':float(accuracy_score(truth,predicted)),
        'confusion_matrix':confusion_matrix(truth,predicted,labels=[0,1,2,3]).tolist(),
        'per_class':[{ 'stage':i, 'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),'support':int(support[i])} for i in range(4)],
        'rul_mae_hours':float(np.abs(error).mean()), 'rul_rmse_hours':float(np.sqrt((error**2).mean())),
        'rul_mape_excluding_under_15min':float((np.abs(error[eligible])/actual_rul[eligible]).mean()),
        'rul_interval_coverage':float(((actual_rul >= rul[:,0].numpy()) & (actual_rul <= rul[:,2].numpy())).mean()),
        'nominal_interval_coverage':0.8, 'parameter_count':sum(p.numel() for p in model.parameters()),
        'inference_median_ms':float(np.median(samples)), 'inference_p95_ms':float(np.percentile(samples,95)),
        'latency_scope':'CPU batch=1 model forward only; excludes feature extraction and IO',
        'healthy_false_positive_rate':float((predicted[truth==0]>0).mean()),
        'critical_fault_recall':float((predicted[truth==3]==3).mean()),
    }


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=12);parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args();torch.set_num_threads(2)
    folder=Path('artifacts');folder.mkdir(exist_ok=True)
    data=simulate_fleet(seed=args.seed);data.to_csv(folder/'simulated_fleet.csv',index=False)
    splits=grouped_split(data,args.seed); baseline=ConditionBaseline().fit(splits['train'])
    experiments={}
    configurations=[('hybrid','hybrid',True,True),('lstm','lstm',True,True),('transformer','transformer',True,True),('no_condition_correction','hybrid',False,True),('fixed_fusion','hybrid',True,False)]
    for name,kind,corrected,gate in configurations:
        start=time.perf_counter()
        model,history=train_one(splits['train'],splits['validation'],baseline,kind,corrected,gate,args.epochs,args.seed)
        metrics=evaluate(model,splits['test'],baseline,corrected)
        metrics['validation_loss_history']=history;metrics['training_seconds']=time.perf_counter()-start
        experiments[name]=metrics
        print(name,json.dumps({k:metrics[k] for k in ['accuracy','rul_mae_hours','inference_p95_ms']},ensure_ascii=False),flush=True)
        if name=='hybrid':
            bundle=folder/'model';bundle.mkdir(exist_ok=True)
            torch.save(model.state_dict(),bundle/'weights.pt')
            manifest={'version':f'synthetic-hybrid-seed{args.seed}-v1','kind':kind,'corrected':corrected,'quality_gate':gate,
                      'sequence_length':16,'window_aggregation_minutes':1,'input_dimension':32,'baseline':baseline.to_dict(),
                      'source':'synthetic','rul_unit':'effective_operating_hours','failure_definition':'simulated wear >= 1',
                      'supported_mode':'drilling','class_labels':['healthy','mild','moderate','severe'],'runtime':'Python 3.12, PyTorch 2.6 CPU',
                      'limitations':['not validated on real tools','intervals not calibrated on real data','no RUL with missing critical sensors']}
            import hashlib
            manifest['weights_sha256']=hashlib.sha256((bundle/'weights.pt').read_bytes()).hexdigest()
            (bundle/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    # Dynamic LSTM-only quantization: backend-supported recurrent weights.
    from torch.ao.quantization import quantize_dynamic
    import io
    primary=MaintenanceModel('hybrid');primary.load_state_dict(torch.load(folder/'model/weights.pt',weights_only=True));primary.eval()
    quantized=quantize_dynamic(copy.deepcopy(primary), {torch.nn.LSTM}, dtype=torch.qint8)
    experiments['hybrid_lstm_int8']=evaluate(quantized,splits['test'],baseline,True)
    experiments['hybrid_lstm_int8']['parameter_count']=sum(p.numel() for p in primary.parameters())
    experiments['hybrid_lstm_int8']['quantization_scope']='LSTM weights only; packed weights do not reduce architecture parameter count'
    for name,m in [('hybrid',primary),('hybrid_lstm_int8',quantized)]:
        buffer=io.BytesIO();torch.save(m.state_dict(),buffer)
        experiments[name]['serialized_weights_bytes']=len(buffer.getvalue())
    # A conventional scalar vibration threshold chosen only on validation.
    candidates=np.linspace(1,6,100)
    val=splits['validation'].dropna(subset=['vibration_rms']);test=splits['test'].dropna(subset=['vibration_rms'])
    threshold=max(candidates,key=lambda threshold:((val.vibration_rms>threshold)==(val.stage>0)).mean())
    experiments['vibration_threshold']={'threshold':float(threshold),'binary_accuracy':float(((test.vibration_rms>threshold)==(test.stage>0)).mean()),'healthy_false_positive_rate':float((test.loc[test.stage==0,'vibration_rms']>threshold).mean()),'fault_recall':float((test.loc[test.stage>0,'vibration_rms']>threshold).mean()),'task':'binary healthy vs any wear; not directly comparable to four-class accuracy'}
    report={'seed':args.seed,'epochs':args.epochs,'source':'synthetic feature trajectories','total_windows':len(data),
            'split_devices':{k:sorted(v.device_id.unique().tolist()) for k,v in splits.items()},'experiments':experiments,
            'limitations':['one seed only','test set contains 3 independent devices','generated feature summaries, not real acquired waveforms','CPU forward latency is not hardware end-to-end latency','no measured power or peak RAM']}
    (folder/'experiment_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
