def draw_scores(mean1, mean2, mean3, std1, std2, std3):
    
    labels = ['precision', 'recall', 'F1', 'SR']

    x = np.arange(len(labels))  # the label locations
    y = np.arange(0, 1, 0.1)
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 8))#可调整柱子粗细高低
    rects1 = ax.bar(x - width, mean1, width, alpha=0.1, color='r' , yerr=std1, label='mean1')
    rects2 = ax.bar(x, mean2, width, alpha=0.1, color='b', yerr=std2, label='mean_paper')
    rects3 = ax.bar(x + width, mean3, width, alpha=0.1, color='g', yerr=std3, label='mean_hodina')
    
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores by three recommendations (20% train data)')
    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.set_xticklabels(labels)
    ax.legend(loc=(0.65, 0.9))
    
    for rect in rects1:
        height = rect.get_height()
        ax.annotate('{:.3f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
    for rect in rects2:
        height = rect.get_height()
        ax.annotate('{:.3f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
    for rect in rects3:
        height = rect.get_height()
        ax.annotate('{:.3f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    fig.tight_layout()
    plt.savefig("scores_by_dif_method_20.jpg")

