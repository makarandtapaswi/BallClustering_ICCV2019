# Video Face Clustering (ICCV 2019)

<strong>Video Face Clustering with Unknown Number of Clusters</strong>  
M. Tapaswi, M. T. Law, and S. Fidler  
International Conference on Computer Vision (ICCV), October 2019.  
[arXiv - coming soon]()

+ Realistic setting for clustering face tracks in videos
+ Number of clusters is not known
+ Background character face tracks are not removed and need to be resolved
+ <strong>Ball Cluster Learning</strong>: a new loss function that carves feature space into balls of a learned radius that can be used as a stopping criterion of agglomerative clustering

---

### Dataset
We use 6 episodes of season 1 of <em>The Big Bang Theory</em> and 6 episodes of season 5 of <em>Buffy - The Vampire Slayer</em>. Face track labels are resolved between background characters.

Original tracks were provided by:  
M. Bäuml, et al. [Semi-supervised Learning with Constraints for Person Identification in Multimedia Data](http://www.cs.toronto.edu/~makarand/papers/CVPR2013.pdf). CVPR 2013.

Please use the <code>download.sh</code> script inside <code>data/</code> for convenience.  
[Face tracks](http://www.cs.toronto.edu/~makarand/downloads/bcl_iccv2019/tracks.tar.gz) (5.3 MB)  
[VGG Face SE-ResNet50-256 features](http://www.cs.toronto.edu/~makarand/downloads/bcl_iccv2019/features.tar.gz) (519 MB)


---

### Evaluation
(02.09.2019) Our final checkpoint with evaluation code has been released.
Numbers can be reproduced by downloading the features and labels and running with `video_name` as one of the following videos: `bbt_s01e01..06` or `buffy_s05e01..06`
```
python evaluate.py <video_name>
```

---

### Code
coming soon

