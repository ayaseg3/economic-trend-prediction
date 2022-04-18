# -*- coding: utf-8 -*-
import torch
import pytorch_lightning as pl
from transformers import BertJapaneseTokenizer, BertModel
import neologdn
import mojimoji
import unicodedata
import re

def preprocessing(data):
    # 半角スペース削除
    data = [x.replace(' ', '') for x in data]
    # 小文字化・unicode正規化・neolog正規化
    data = [unicodedata.normalize('NFKC', neologdn.normalize(x)).lower() for x in data]
    # 全角から半角
    data = [mojimoji.zen_to_han(x) for x in data]
    # 半角記号、全角記号の置換
    data = [re.sub(u'[■-♯]', '', re.sub(r'[!-/:-@[-`{-~]', '', x)) for x in data]
    # 手動置換
    data = [re.sub(r'[｢｣､]', '', x) for x in data]
    return data

def split_punc(data):
    a = [x.split('｡') for x in data]
    return a

def create_sentence(data):
    data = split_punc(preprocessing(data))
    return data

def list_max(lis):
        if lis == [[0, 0, 0]]:
            return 3
        elif lis == [[1, 0, 0]]:
            return 5
        elif lis == [[1, 1, 0]]:
            return 4
        elif lis == [[0, 1, 0]]:
            return 3
        elif lis == [[0, 1, 1]]:
            return 2
        elif lis == [[0, 0, 1]]:
            return 1
        else:
            print('error')
            return 3

class BertForSequenceClassificationMultiLabel(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        # BertModelのロード
        self.bert = BertModel.from_pretrained(model_name)
        # 線形変換を初期化しておく
        self.linear = torch.nn.Linear(
            self.bert.config.hidden_size, num_labels
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None
    ):
        # データを入力しBERTの最終層の出力を得る。
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)
        last_hidden_state = bert_output.last_hidden_state

        # [PAD]以外のトークンで隠れ状態の平均をとる
        averaged_hidden_state = \
            (last_hidden_state*attention_mask.unsqueeze(-1)).sum(1) \
            / attention_mask.sum(1, keepdim=True)

        # 線形変換
        scores = self.linear(averaged_hidden_state)

        # 出力の形式を整える。
        output = {'logits': scores}

        # labelsが入力に含まれていたら、損失を計算し出力する。
        if labels is not None:
            loss = torch.nn.BCEWithLogitsLoss()(scores, labels.float())
            output['loss'] = loss

        # 属性でアクセスできるようにする。
        output = type('bert_output', (object,), output)

        return output

class BertForSequenceClassificationMultiLabel_pl(pl.LightningModule):

    def __init__(self, model_name, num_labels, lr):
        super().__init__()
        self.save_hyperparameters()
        self.bert_scml = BertForSequenceClassificationMultiLabel(
            model_name, num_labels=num_labels
        )

    def training_step(self, batch, batch_idx):
        output = self.bert_scml(**batch)
        loss = output.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.bert_scml(**batch)
        val_loss = output.loss
        self.log('val_loss', val_loss)

    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        output = self.bert_scml(**batch)
        scores = output.logits
        labels_predicted = ( scores > 0 ).int()
        num_correct = ( labels_predicted == labels ).all(-1).sum().item()
        accuracy = num_correct/scores.size(0)
        self.log('accuracy', accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def token_embedding_multi(text_list, model, tokenizer):
        # 文章の符号化
        encoding = tokenizer(
            text_list,
            max_length=100,
            padding='max_length', # 最長に合わせる
            truncation=True,
            return_tensors='pt'
        )
        # データをGPUに載せる
        encoding = { k: v.to(device) for k, v in encoding.items() }
        # BERTでの処理
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            output = model(**encoding)
        scores = output.logits
        labels_predicted = (scores > 0).int().cpu().numpy().tolist()
        return labels_predicted

# sents : 景気予測文章のリスト
'''
# 2022年12月予想 # -> 3.000
sents = [
        '12月13日に公表される日銀短観（2021年12月調査）の業況判断DI（最近）は、大企業製造業で、前回調査（2021年9月調査）から2ポイント改善の20と予測する。部品不足が解消した自動車などで目立って改善するも、全体ではコロナ前を上回る水準まで回復しており、改善は小幅にとどまろう。先行きは、市況上昇が一部業種の利益を下押しし、2ポイント改善の22となろう。大企業非製造業の業況判断DI（最近）は、前回調査から6ポイント改善の8と予測する。緊急事態宣言が全面解除され、厳しい行動制限下で需要が激減していた対面型サービスを中心に大幅に改善しよう。先行きは、感染状況が落ち着く中、経済活動の正常化が進むことへの期待が高まり、7ポイント改善の15となろ。',
        '製造業の業況判断 DI（大企業）は、+16%ポイント（前回調査から 2%ポイント低下）と 6 期ぶりの悪化を予測する。円安進行などによる収益環境の改善が業況改善要因となるものの、半導体不足、ASEAN での感染拡大による部品供給の停滞、原材料価格の上昇、財消費の特需一服などが生産や企業収益の下押し要因となり、製造業全体としては業況悪化を予想する。非製造業の業況判断 DI（大企業）は、+4%ポイント（前回調査から 2％ポイント上昇）と小幅ながらも 6 期連続の改善を予測する。緊急事態宣言が解除された 10 月以降も新規感染者は低位で推移しており、外出行動は持ち直している。宿泊・飲食サービス、対個人サービスなど外出関連業種を中心に業況改善を予測する。先行きの業況判断 DI（大企業）は、製造業が+16%ポイント（12 月時点の業況判断から変化なし）、非製造業は+5%ポイント（12 月時点の業況判断から 1％ポイント上昇）と予測する。外出行動が持ち直すなかで消費の回復が見込まれる。一方、新型コロナウイルスの感染拡大による世界の経済活動への悪影響は、引き続き先行きの懸念材料となる。ASEAN の一部の国では感染が再拡大しており、部品不足の解消時期には不透明感が残る。感染力の強い変異種の出現によって各国で防疫措置が強まり、世界的に経済活動の抑制度合いが強まる可能性もある。このため、先行きの業況判断はほぼ横ばいにとどまるとみる。',
        '2021 年 12 月 13 日に公表予定の 12 月日銀短観において、大企業製造業の業況判断 DI（最近）は＋19％pt（前回調査からの変化幅：＋1％pt）、大企業非製造業では＋4％pt（同：＋2％pt）を予想する（図表 1）。業況判断 DI の水準は、既にコロナ禍前の水準を回復している製造業と低迷が続く非製造業の間に大きな差があり引き続き業種間格差が見られるものの、方向感では両者とも小幅ながらも改善が見込まれる。後述する業況判断 DI（先行き）と併せて評価すると、世界的な財需要への強さにけん引された製造業中心の回復局面から、経済活動再開の本格化を契機とした非製造業中心の回復局面へと移行していくことへの期待感が示されよう。大企業製造業の内訳を見ると、「自動車」の業況判断 DI が改善すると予想する。これまで業況を悪化させてきた、半導体不足や部品供給元である東南アジアでの新型コロナウイルス感染症の拡大を受けた生産抑制の悪影響は 12 月に入り概ね解消に向かっているとみられる。ただし、過去の自動車減産の影響が残存したことに加え、原材料価格上昇の影響もあり、「金属製品」、「鉄鋼」といった業種の業況判断 DI は小幅な悪化を予想する。このほか、小売店向けを中心とした受注の回復が、「繊維」の業況判断 DI の改善に寄与したほか、堅調な半導体需要が「電気機械」や「化学」を押し上げたとみている。大企業非製造業に関しては、緊急事態宣言等の全面解除が「対個人サービス」、「宿泊・飲食サービス」、「小売」といった業種の業況判断 DI の改善に寄与した可能性が高い。加えて、「小売」に関しては、挽回消費の発現も業況判断 DI を押し上げる働きをしたとみられる。ただし、「対個人サービス」、「宿泊・飲食サービス」に関しては水準で見ると依然として低位にとどまっており、本格的な回復にはなお時間を要するとみられる。このほか「卸売」では、資源価格の上昇や物流の拡大が収益の改善に寄与したと予想する。'
    ]

# 2022年3月予想 # -> 3.178
sents = [
        '4月1日に公表される日銀短観（2022年3月調査）の業況判断DI（最近）は、大企業製造業で、前回調査（2021年12月調査）から3ポイント悪化の15と予測する。資源価格高騰と円安による企業のコスト負担の増加、部品不足による自動車の減産などが業況を下押ししたとみられる。先行きは、ウクライナ危機に伴う資源価格の更なる上昇や海外経済悪化への警戒感が幅広い業種の業況を悪化させ、2ポイント悪化の13となろう。',
        '大企業非製造業の業況判断DI（最近）は、前回調査から3ポイント悪化の6と予測する。昨年12月以降の新型コロナ・オミクロン株の流行を受けて、対面型サービスを中心に悪化しよう。先行きは、感染状況が落ち着き、経済活動正常化への期待が高まり、1ポイント改善の7となろう。',
        '2021年度の大企業設備投資計画は、例年のパターン通り、製造業は前回調査から小幅下方修正、非製造業は大きな修正はない見込み。2022年度の設備投資は、DX推進、環境規制への対応、ウィズ／アフターコロナへの適応など、景気に左右されづらい投資を中心に、特に製造業において増加の計画が示されよう。',
        '製造業の業況判断 DI（大企業）は、前回１２月調査（調査対象企業見直し後の新ベース）から▲8％ポイント低下の+9%ポイントと、２０２０年６月調査以来の悪化を予測する。既往のエネルギー価格の上昇や部品・半導体などの供給制約に加え、ロシアによるウクライナ侵攻の悪影響が幅広い業種で広がっている。①エネルギー・鉱物資源・食料などの資源価格の一段の上昇や供給制約、②物流遅延や運賃上昇、③ロシア向け輸出やロシアからの調達の制約、④ロシアでの事業・生産の停止、などが業況悪化要因となろう。また、中国での感染急拡大による悪影響も出始めている。非製造業の業況判断 DI（大企業）は、前回１２月調査（調査対象企業見直し後の新ベース）から▲5％ポイント低下の+5%ポイントと予測する。国内のオミクロン変異株の流行により、前回１２月調査時点と比べて外出関連を中心に消費が抑制されており、幅広い業種の業況悪化要因となる。また、電気・ガスや運輸・郵便ではエネルギー価格など投入コスト上昇による収益悪化、卸売ではロシア事業の不透明感の強まりが、業況の悪化要因となろう。'
        '4月1日公表予定の日銀短観（3月調査）では、大企業・製造業の業況判断ＤＩが＋13％Pt（12月調査：＋17％Pt）と、前回調査（調査対象見直し後の新ベース、以下同）から4ポイント悪化を予測する（図表1）。製造業は、半導体不足や感染拡大による自動車減産、資源価格高騰に起因する原材料・燃料コストの上昇を背景に、業況は悪化するだろう。 ',
        '素材業種は、原材料価格の上昇によるコスト増を背景に鉄鋼や非鉄金属、化学等の業況悪化が予想される。加工業種については、特に自動車の悪化が見込まれる。半導体不足の長期化、オミクロン株感染急拡大による工場稼働停止を受けて、1月の自動車生産は大幅に減少した。また、原材料価格上昇の悪影響は加工業種全般に波及し始めている。1月の投入・産出物価指数では、加工業種の投入物価が前年比＋7.9％まで上昇した一方、産出物価は同＋3.3％にとどまった（図表2）。多くの企業は原材料費の高騰を十分に価格転嫁できず、収益が圧迫されているとみられる。大企業・非製造業の業況判断ＤＩは＋7％Pt（12月調査：＋10％Pt）と、前回調査から3ポイントの悪化を予測する。1月後半以降のオミクロン株感染急拡大を受けて、サービス消費は大幅に落ち込んだ。2月に感染はピークアウトしたが、感染者数の減少ペースは緩慢であり、宿泊・飲食サービスや対個人サービスの業況悪化が見込まれる。また、資材価格の上昇を受けて、建設は悪化するだろう。一方、資源価格上昇の恩恵を受ける形で、卸売の業況は改善するとみている。'
    ]
'''

if __name__ == '__main__':
    torch.manual_seed(10)
    if torch.cuda.is_available() == True:
        torch.cuda.manual_seed(10)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)

    best_model_path = './epoch=7-step=303.ckpt'
    model = BertForSequenceClassificationMultiLabel_pl.load_from_checkpoint(best_model_path)
    bert_scml = model.bert_scml.to(device)

    ## 手動入力の場合 ##
    #sents = create_sentence(sents)
    
    ## コマンド入力の場合 ##
    data = input("景気予測文章の入力:")
    sents = create_sentence([data])
    #sents_2 = [[y for y in x if len(y) > 0] for x in sents]
    data = []
    for x in sents:
        for y in x:
            if len(y) > 0:
                data.append(y)

    with torch.no_grad():
        sum = 0
        cnt = 0
        print('-'*30)
        for sent in data:
            out = token_embedding_multi(sent, bert_scml, tokenizer)
            print('sentence :', sent)
            print('sent_pred :', list_max(out))
            print('-'*30)
            sum += list_max(out)
            cnt += 1
        ave = sum / cnt
    print('='*30)
    print('pred_range : [1, 2, 3, 4, 5]')
    print('prediction :', ave)
