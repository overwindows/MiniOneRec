import os
import pickle
import pandas as pd
import numpy as np
import tqdm


class NewsSelectorTest(object):
    # popular bucket (clicked): [key/ value] = [bucket index/ list of news]
    # popular bucket (recommended): [key/ value] = [bucket index/ list of news]
    # sorted list of newses in order of time. (Only containing new keys)

    def __init__(self, data_type1, data_type2, num_pop, num_fresh, bucket_size=3):
        # data_type1: 'demo', 'large'
        # data_type2: 'dev', 'train'
        self.bucket_size = bucket_size # size of buckets
        self.num_pop = num_pop
        self.num_fresh = num_fresh

        # Define file names 
        self.data_path = '/data/mind/' + 'MIND' + data_type1 + '_' + data_type2 + '/'
        self.behavior_file = self.data_path + 'behaviors.tsv'
        # self.news_file = self.data_path + 'integrated_news.tsv'  # need combined version
        # self.news_file = '/home/yuna/data/msn_' + data_type2 + '.csv'

        self.fresh_news_df_file = self.data_path + "fresh_news_df.pickle"
        self.pop_hash_clicked_file = self.data_path + "pop_hash_clicked_{}.pickle".format(self.bucket_size)
        self.pop_hash_recommended_file = self.data_path + "pop_hash_recommended_{}.pickle".format(self.bucket_size)

        # Popular news and sorted news list
        self.pop_hash_clicked = None # [key/ value] = [bucket index/ list of news]
        self.pop_hash_recommended = None # [key/ value] = [bucket index/ list of news]
        self.fresh_news_df = None # sorted list of news ids
        self.sorted_dates = None

    def get_fresh(self, time):
        # input:  time - query time
        #         k - number of news
        # output: list of fresh news IDs

        # Load or generate sorted newslist file
        if self.fresh_news_df is None:
            self.load_fresh_news_df()
        # get list of fresh news
        # _df = self.fresh_news_df[self.fresh_news_df['date'] <= self.parsing_behavior_time(time, return_type='int')]
        # res = _df.sort_values(by='date', ascending=False, inplace=False).iloc[0:self.num_fresh]

        query = self.parsing_behavior_time(time, return_type='int')
        start_idx = self.sorted_dates.size - np.searchsorted(self.sorted_dates[::-1], query, side='right')
        res = self.fresh_news_df.iloc[start_idx:start_idx+self.num_fresh]

        # print("query time: ", self.parsing_behavior_time(time, return_type='int'))
        # print("results:")
        # for i in (res['date'].values):
        #     print(i, "--> query - i = ", self.parsing_behavior_time(time, return_type='int')-i )
        return res.iloc[:, 1].values

    def _decrease_bucket_key(self, bucket_key):
        # key: (month, date, year, time_bucket_index)
        _m_d_dict = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30,
                     7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
        if bucket_key[3] == 0:
            if bucket_key[1] == 1:
                if bucket_key[0] == 1:
                    new_key = (12, _m_d_dict[bucket_key[0] - 1], bucket_key[2] - 1, 23 // self.bucket_size)
                else:
                    new_key = (bucket_key[0] - 1, _m_d_dict[bucket_key[0] - 1], bucket_key[2], 23 // self.bucket_size)
            else:
                new_key = (bucket_key[0], bucket_key[1] - 1, bucket_key[2], 23 // self.bucket_size)
        else:
            new_key = (bucket_key[0], bucket_key[1], bucket_key[2], bucket_key[3] - 1)
        return new_key

    def get_pop_clicked(self, time):
        if self.pop_hash_clicked is None:
            self.load_pop_hash_clicked()
        ltime = self.parsing_behavior_time(time)
        bucket_key = (ltime[0], ltime[1], ltime[2], ltime[3] // self.bucket_size)

        while bucket_key not in self.pop_hash_clicked.keys():
            bucket_key = self._decrease_bucket_key(bucket_key)

        sorted_news_list = sorted(self.pop_hash_clicked[bucket_key].items(),
                                  reverse=True, key=lambda item: item[1])
        news_list = sorted_news_list[0:self.num_pop]
        res = [item[0] for item in news_list]
        return res

    def get_pop_recommended(self, time):
        if self.pop_hash_recommended is None:
            self.load_pop_hash_recommended()
        ltime = self.parsing_behavior_time(time)
        bucket_key = (ltime[0], ltime[1], ltime[2], ltime[3] // self.bucket_size)

        while bucket_key not in self.pop_hash_recommended.keys():
            bucket_key = self._decrease_bucket_key(bucket_key)

        sorted_news_list = sorted(self.pop_hash_recommended[bucket_key].items(),
                                  reverse=True, key=lambda item: item[1])
        news_list = sorted_news_list[0:self.num_pop]
        res = [item[0] for item in news_list]
        return res

    def load_fresh_news_df(self):
        if os.path.isfile(self.fresh_news_df_file):
            with open(self.fresh_news_df_file, 'rb') as f:
                self.fresh_news_df = pickle.load(f)
        else:
            self.generate_fresh_news_df()

        self.sorted_dates = self.fresh_news_df['date'].to_numpy()

    def load_pop_hash_clicked(self):
        if os.path.isfile(self.pop_hash_clicked_file):
            with open(self.pop_hash_clicked_file, 'rb') as f:
                self.pop_hash_clicked = pickle.load(f)
        else:
            self.generate_pop_hash_clicked()

    def load_pop_hash_recommended(self):
        if os.path.isfile(self.pop_hash_recommended_file):
            with open(self.pop_hash_recommended_file, 'rb') as f:
                self.pop_hash_recommended = pickle.load(f)
        else:
            self.generate_pop_hash_recommended()

    def generate_fresh_news_df(self):
        news2time = {}
        df = pd.read_csv(self.behavior_file, sep='\t', header=None)

        for i in tqdm.tqdm(range(df.shape[0])):
            _imp = df.iloc[i, :]
            ltime = self._process_news_date2(_imp[2])
            imprs = _imp[4].split(' ')
            for imp in imprs:
                if imp in news2time:
                    news2time[imp].append(ltime)
                else:
                    news2time[imp] = [ltime]
        for k, v in tqdm.tqdm(news2time.items()):
            news2time[k] = min(v)
        _df = pd.DataFrame(news2time.items(), columns=['nid', 'date'])[['date', 'nid']]
        _df.sort_values(by='date', ascending=False, inplace=True)

        self.fresh_news_df = _df
        with open(self.fresh_news_df_file, 'wb') as f:
            pickle.dump(_df, f)

    def generate_pop_hash_clicked(self):
        df = pd.read_csv(self.behavior_file, sep='\t', header=None)
        buckets = {}
        for i in range(df.shape[0]):
            _imp = df.iloc[i,:]
            ltime = self.parsing_behavior_time(_imp[2])
            clicked_imps = self.get_news_list(_imp[4], all=False)
            bucket_key = (ltime[0], ltime[1], ltime[2], ltime[3]//self.bucket_size)

            if bucket_key in buckets.keys():
                _d = buckets[bucket_key]
                for _imp in clicked_imps:
                    if _imp in _d.keys():
                        _d[_imp] += 1
                    else:
                        _d[_imp] = 1
            else:
                buckets[bucket_key] = {}
                for _imp in clicked_imps:
                    buckets[bucket_key][_imp] = 1
        # Save bucket as a pickle file
        with open(self.pop_hash_clicked_file, 'wb') as f:
            pickle.dump(buckets, f)
        self.pop_hash_clicked = buckets

    def generate_pop_hash_recommended(self):
        df = pd.read_csv(self.behavior_file, sep='\t', header=None)
        buckets = {}
        for i in range(df.shape[0]):
            _imp = df.iloc[i, :]
            ltime = self.parsing_behavior_time(_imp[2])
            recommended_imps = self.get_news_list(_imp[4], all=True)
            bucket_key = (ltime[0], ltime[1], ltime[2], ltime[3] // self.bucket_size)

            if bucket_key in buckets.keys():
                _d = buckets[bucket_key]
                for _imp in recommended_imps:
                    if _imp in _d.keys():
                        _d[_imp] += 1
                    else:
                        _d[_imp] = 1
            else:
                buckets[bucket_key] = {}
                for _imp in recommended_imps:
                    buckets[bucket_key][_imp] = 1

        # Save bucket as a pickle file
        with open(self.pop_hash_recommended_file, 'wb') as f:
            pickle.dump(buckets, f)

        self.pop_hash_recommended = buckets

    def parsing_behavior_time(self, ftime, return_type='list'):
        _mdy = ftime.split(' ')[0].split('/')
        _hmt = ftime.split(' ')[1].split(':')
        if ftime.split(' ')[2] == 'PM':
            _hmt[0] = int(_hmt[0])
            _hmt[0] += 12
        ltime = [int(_mdy[0]), int(_mdy[1]), int(_mdy[2]),
                 int(_hmt[0]), int(_hmt[1]), int(_hmt[2])]
        if return_type == 'list':
            return ltime
        elif return_type == 'int':
            s = '{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(
                ltime[2], ltime[0], ltime[1], ltime[3], ltime[4], ltime[5])
            return int(s)
        else:
            print("Return type error")
            return None

    def get_news_list(self, raw_imp, all=False):
        _imps = raw_imp.split(' ')
        if all:
            _news_list = _imps
        else:
            _news_list = [_i[:-2] for _i in _imps if _i[-1] == '1']
        return _news_list

    def _process_news_date(self, date):
        # need to consider nan values => 1e15 !
        if type(date) == float:
            return 1e15
        print(date)
        t1 = date.split('T')[0].split('-')
        t2 = date.split('T')[1][:-1].split(':')
        t2[2] = t2[2][:2]
        res = t1[0] + t1[1] + t1[2] + t2[0] + t2[1] + t2[2]
        return int(res)

    def _process_news_date2(self, date):
        # need to consider nan values => 1e15 !
        if type(date) == float:
            return 1e15
        t1 = date.split(' ')[0].split('/')
        t2 = date.split(' ')[1].split(':')
        t3 = date.split(' ')[2]
        if t3 == 'PM':
            t2[0] = str(int(t2[0]) + 12)
        res = t1[2] + t1[1] + t1[0] + t2[0] + t2[1] + t2[2]
        return int(res)


if __name__ == "__main__":
    nr = NewsSelectorTest(num_pop=20, num_fresh=20, data_type1='large', data_type2='test')
    ns1 = nr.get_pop_recommended('11/19/2019 11:37:45 AM')
    print(ns1)
    # ns2 = nr.get_pop_clicked('11/15/2019 8:55:22 AM')
    ns3 = nr.get_fresh('11/15/2019 8:55:22 AM')

    print(ns3)
