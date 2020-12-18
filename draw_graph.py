import matplotlib.pyplot as plt
import csv

def read_csv(file_name):
    metrics = [[0,0]]
    with open(file_name + ".csv", 'r', encoding='utf-8') as f:
        rdr = csv.reader(f)
        for line in rdr:
            metrics.append(line)
    
    return metrics

def draw(metrics, label, mode='acc'):
    if mode == 'acc':
        acc = [float(m[1]) for m in metrics]
        plt.plot(range(len(acc)), acc, label=label)

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        
    elif mode == 'loss':
        loss = [float(m[0]) for m in metrics]
        plt.plot(range(len(loss)), loss, label=label)
    
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='lower right')

    elif mode == 'data_size':
        yl = ["FedAvg (C=0.1)", "FedAvg (C=0.2)", "FedAvg (C=0.5)", "FedAvg (C=1.0)", 'FedPSO (C=1.0)']
        ds = [0.1, 0.2, 0.5, 1, 0]
        plt.bar(yl, ds)

        ds2 = [0, 0, 0, 0, 0.55]
        plt.bar(yl, ds2)
    
        plt.ylabel(label)
        # plt.legend(loc='lower right')
    

if __name__ == "__main__":
    # file = []
    # file.append("mnist_output_original_FL_C_0.1_LR_0.0025_CLI_10_CLI_EPOCHS_5_TOTAL_EPOCHS_30_BATCH_10")
    # file.append("mnist_output_original_FL_C_0.2_LR_0.0025_CLI_10_CLI_EPOCHS_5_TOTAL_EPOCHS_30_BATCH_10")
    # file.append("mnist_output_original_FL_C_0.5_LR_0.0025_CLI_10_CLI_EPOCHS_5_TOTAL_EPOCHS_30_BATCH_10")
    # file.append("mnist_output_original_FL_C_1_LR_0.0025_CLI_10_CLI_EPOCHS_5_TOTAL_EPOCHS_30_BATCH_10")
    # file.append("mnist_output_PSO_FL_LR_0.0025_CLI_10_CLI_EPOCHS_5_TOTAL_EPOCHS_30_BATCH_10")

    # label = ["C = 0.1", "C = 0.2", "C = 0.5", "C = 1.0"]
    # for i in range(4):
    #     metrics = read_csv(file[i])
    #     draw(metrics, label[i], mode="acc")
    
    # metrics = read_csv(file[4])
    # draw(metrics, "FedPSO", mode="acc")

    # plt.savefig("mnist_acc.png")
    # plt.clf()

    draw(None, "Transmission Cost", mode="data_size")
    plt.savefig("data.png")
    plt.clf()