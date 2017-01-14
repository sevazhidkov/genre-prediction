# Genre Prediction

Predict music genre for the events based on the online data. Made for a [Hi Karl](https://hikarl.com/) interview.

**Technologies**: Python, Jupyter Notebook, scikit-learn, gensim

**Contributors**: Seva Zhidkov (iam@sevazhidkov.me)

**Initial date**: 14.01.2017

## Task summary

We have a list of different events in NYC. We know basic information about
each event such as artist, cost, description and others. Our goal is to predict
a music genre (or genres) of the event. For example, it could be rock party or
jazz concert.

The most annoying problem was big amount of missing data.
Events list and related information were aggregated from different sites, so
it's hard to set one structure for all data.

One of the most helpful parameter was a description of the event. It usually contains information
about type of the event.

## Technology overview

I created an hybrid model that uses extendable features list as input and
returns genre probabilites vector.

Features includes LDA topic model of the item descriptions, artists and their genres,
cost etc.

I focused on extensibility and speed. It's very important things in the production Data Science.
We can easily create a new feature if it's necessary and retrain model.
Also my code doesn't require any structure changes for creating model.

## Notes on the LDA model

LDA model is trained on events descriptions and creates 20 topics:

```
[(0, '0.026*"door" + 0.019*"us" + 0.016*"show" + 0.011*"open" + 0.008*"read"'),
 (1,
  '0.017*"music" + 0.017*"jazz" + 0.014*"latin" + 0.013*"perform" + 0.012*"new"'),
 (2,
  '0.023*"show" + 0.011*"amateur" + 0.010*"get" + 0.009*"29" + 0.008*"bunni"'),
 (3,
  '0.024*"jazz" + 0.023*"music" + 0.019*"compo" + 0.012*"new" + 0.010*"perform"'),
 (4,
  '0.009*"music" + 0.008*"new" + 0.008*"relea" + 0.008*"band" + 0.006*"album"'),
 (5,
  '0.023*"ticket" + 0.022*"30pm" + 0.019*"will" + 0.018*"door" + 0.016*"00pm"'),
 (6,
  '0.022*"tax" + 0.021*"will" + 0.018*"brunch" + 0.017*"applic" + 0.016*"gratuiti"'),
 (7,
  '0.018*"music" + 0.010*"new" + 0.009*"record" + 0.008*"perform" + 0.007*"album"'),
 (8,
  '0.010*"new" + 0.009*"music" + 0.008*"band" + 0.006*"like" + 0.006*"song"'),
 (9,
  '0.030*"blue" + 0.017*"band" + 0.013*"new" + 0.011*"york" + 0.010*"featur"'),
 (10,
  '0.040*"soul" + 0.023*"back" + 0.021*"music" + 0.016*"take" + 0.015*"us"'),
 (11,
  '0.012*"music" + 0.009*"record" + 0.007*"album" + 0.007*"band" + 0.005*"new"'),
 (12,
  '0.041*"princess" + 0.028*"album" + 0.023*"beatl" + 0.015*"stop" + 0.013*"featur"'),
 (13,
  '0.015*"music" + 0.010*"night" + 0.006*"record" + 0.005*"show" + 0.005*"late"'),
 (14,
  '0.017*"fairi" + 0.009*"music" + 0.008*"parti" + 0.008*"show" + 0.006*"toni"'),
 (15,
  '0.009*"music" + 0.009*"record" + 0.008*"new" + 0.007*"band" + 0.006*"album"'),
 (16,
  '0.025*"new" + 0.017*"life" + 0.016*"tale" + 0.015*"emperor" + 0.012*"relea"'),
 (17,
  '0.013*"v" + 0.011*"women" + 0.007*"youtub" + 0.007*"ymusic" + 0.007*"door"'),
 (18,
  '0.009*"like" + 0.008*"love" + 0.008*"say" + 0.008*"song" + 0.007*"music"'),
 (19,
  '0.015*"band" + 0.009*"record" + 0.009*"album" + 0.008*"music" + 0.007*"song"')]
```

It looks pretty well. Words is stemmed using Snowball stemmer by nltk and filtered
by small list of stop words.

## The model

Final (for the interview) model is Linear Support Vector Classification. It's very good for production
because of high speed and low memory usage.

Model uses 1860 features generated from each event data. I use one Linear SVC model for each genre.

## Metrics

**Rock classification**:

| Accuracy score | 0.83064516129  |
|----------------|----------------|
| F1 score       | 0.886486486486 |
| ROC AUC        | 0.841179197182 |

ROC curve:

![ROC curve](http://i.imgur.com/raZnbey.png)

**All genres classification**:

ROC curve:

![ROC curves](http://i.imgur.com/N1BQhlS.png)
