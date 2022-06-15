# Implementation on Super-Resolution
Implementation on previous methods and various models will be available soon.

## Useage

```bash
cd super-resolution/src
```

For training,
```bash
bash train_sls_carnX4.sh $gpu $target_budget  # Training on DIV2K
```

For test,
```bash
bash test_sls_carnX4.sh $gpu $exp_name    # Test on Set14, B100, Urban100
```

To see the computational costs (w.r.t MACs and Num. Params.) of a trained model,
```bash
bash compute_costs.sh $gpu $model_dir
```