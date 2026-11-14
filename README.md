# DNLP alfred-instr2action
**자연어 지시문 기반 로봇 행동 계획 생성: HMM과 Transformer, Action Chunking 비교**

※실제 로봇이나 시뮬레이터를 사용하지 않고, 고수준 계획만 offline으로 평가하여 **로봇 행동 계획** 생성

## Dataset
**사용 데이터셋** : [ALFRAD](https://github.com/askforalfred/alfred/tree/master/data)(Lite 35MB):

```
$ cd $ALFRED_ROOT/data
$ sh download_data.sh json
```
본 프로젝트에서는 **지시문**과 **행동 시퀀스**만 사용.

