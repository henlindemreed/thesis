import unittest
import copy
from Framework import *

class DataTest(unittest.TestCase):

    def test_sysconf(self):
        dataset = load_dataset('sysconf')
        self.assertEqual(len(dataset), 2439)
        self.assertTrue('abstr' in dir(dataset[0]))
        self.assertTrue('id' in dir(dataset[0]))
        self.assertTrue('paper' in dir(dataset[0]))
        train, test = partition_dataset(dataset, 0.1)
        self.assertEqual(len(train), 2196)
        self.assertEqual(len(test), 243)


class ModelTest(unittest.TestCase):

    def setUp(self):
        self.dataset = load_dataset('sysconf')
        self.train, self.test = partition_dataset(self.dataset, 0.1)

    def test_nonsense(self):
        model = init_model('nonsense')
        self.assertTrue('train' in dir(model))
        self.assertTrue('generate' in dir(model))
        result = Evaluate(model, self.test)
        rougesum = sum([ sum([ sum([ x for x in r.values()])
                                for r in paper[1].values()])
                                for paper in result])
        self.assertEqual(rougesum, 0)

    def test_cheater(self):
        model = init_model('cheater')
        self.assertTrue('train' in dir(model))
        self.assertTrue('generate' in dir(model))
        result = Evaluate(model, self.test)
        rougesum = sum([ sum([ sum([ x for x in r.values()])
                                for r in paper[1].values()])
                                for paper in result])
        self.assertTrue(rougesum > 2100)

    def test_basic(self):
        model = init_model('basic')
        self.assertTrue('train' in dir(model))
        self.assertTrue('generate' in dir(model))
        model.train(self.train)
        result = Evaluate(model, self.test)
        rougesum = sum([ sum([ sum([ x for x in r.values()])
                                for r in paper[1].values()])
                                for paper in result])
        self.assertTrue(0 < rougesum < 2100)


class GoodModels(unittest.TestCase):

    def setUp(self):
        self.model_names = ['cheater', 'basic']
        self.models = [init_model(name) for name in self.model_names]
        self.dataset = load_dataset('sysconf')
        self.train, self.test = partition_dataset(self.dataset, 0.1)
        for m in self.models:
            Train(m, self.train)

    # Assert that a candidate fits its own document better than any other
    # Test over all models (configurable)
    # Test several times with random connections
    def test_candidate_best_match(self):
        rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True,
                rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)
        for j in range(50):
            target = random.choice(self.test)
            noise = [target] + random.sample(self.dataset, 50)
            for m, n in zip(self.models, self.model_names):
                print(n)
                candidate = m.generate(target)
                results = rouge.evaluate([candidate],[noise[0].abstr])
                orig_results = copy.deepcopy(results)
                for i in range(1,len(noise)):
                    r = rouge.evaluate([candidate],[noise[i].abstr])
                    for k in r:
                        if r[k]['f'] > results[k]['f']:
                            results[k]['f'] = r[k]['f']
                self.assertEqual(results, orig_results)




if __name__ == '__main__':
    unittest.main()