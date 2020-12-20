import matplotlib.pyplot as plt
import csv

def read_csv(file_name):
    metrics = [[0,0]]
    with open(file_name + ".csv", 'r', encoding='utf-8') as f:
        rdr = csv.reader(f)
        for line in rdr:
            metrics.append(line)
    
    return metrics

def draw(metrics, label, color, marker, mode='acc'):
    if mode == 'acc':
        acc = [float(m[1]) for m in metrics]
        plt.plot(range(len(acc)), acc, color=color, linestyle=marker, label=label)

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        # plt.ylim(0, 0.8)
        plt.title('MNIST')
        plt.legend(loc='lower right')
        
    elif mode == 'failure':
        fail = [float(m[1]) for m in metrics]
        plt.plot(range(len(fail)), fail, color=color, linestyle=marker, label=label)

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.ylim(0, 0.8)
        plt.title('Communication Failure Rate - FedAvg ')
        plt.legend(loc='lower right')

    elif mode == 'data_size':
        yl = ['FedPSO', "FedAvg (C=1.0)", "C=0.5", "C=0.2", "C=0.1"]
        ds_pso = [0.55, 0, 0, 0, 0]
        plt.bar(yl, ds_pso, color="darkorange", label="FedPSO")
        
        ds_origin = [0, 1.0, 0.5, 0.2, 0.1]
        plt.bar(yl, ds_origin, color="dimgray", label="FedAvg")
    
        plt.ylabel(label)
        plt.legend(loc='upper right')
    
def acc_graph():
    file = []
    # file.append("best_score_70_output_PSO_FL_LR_0.0025_CLI_5_CLI_EPOCHS_5_TOTAL_EPOCHS_30_BATCH_10")
    # file.append("output_original_FL_C_1.0_LR_0.0025_CLI_10_CLI_EPOCHS_5_TOTAL_EPOCHS_30_BATCH_10")
    # file.append("output_original_FL_C_0.5_LR_0.0025_CLI_10_CLI_EPOCHS_5_TOTAL_EPOCHS_30_BATCH_10")
    # file.append("output_original_FL_C_0.2_LR_0.0025_CLI_10_CLI_EPOCHS_5_TOTAL_EPOCHS_30_BATCH_10")
    # file.append("output_original_FL_C_0.1_LR_0.0025_CLI_10_CLI_EPOCHS_5_TOTAL_EPOCHS_30_BATCH_10")

    file.append("mnist_output/mnist_randomDrop_0%_output_PSO_FL_LR_0.0025_CLI_10_CLI_EPOCHS_5_TOTAL_EPOCHS_30_BATCH_10")
    file.append("mnist_output/mnist_output_original_FL_C_1_LR_0.0025_CLI_10_CLI_EPOCHS_5_TOTAL_EPOCHS_30_BATCH_10")
    file.append("mnist_output/mnist_output_original_FL_C_0.5_LR_0.0025_CLI_10_CLI_EPOCHS_5_TOTAL_EPOCHS_30_BATCH_10")
    file.append("mnist_output/mnist_output_original_FL_C_0.2_LR_0.0025_CLI_10_CLI_EPOCHS_5_TOTAL_EPOCHS_30_BATCH_10")
    file.append("mnist_output/mnist_output_original_FL_C_0.1_LR_0.0025_CLI_10_CLI_EPOCHS_5_TOTAL_EPOCHS_30_BATCH_10")

    
    metrics = read_csv(file[0])
    draw(metrics, "FedPSO", "darkorange", "-", mode="acc")

    label = ["C = 1.0", "C = 0.5", "C = 0.2", "C = 0.1"]
    mark = ["-", "--", "-.", ":"]
    for i in range(4):
        metrics = read_csv(file[i+1])
        draw(metrics, label[i], "dimgray", mark[i], mode="acc")
    

    plt.savefig("mnist_acc.png")
    plt.clf()


def failure():
    file = []
    file.append("origin_drop/output_original_FL_C_1.0_LR_0.0025_CLI_10_CLI_EPOCHS_5_TOTAL_EPOCHS_30_BATCH_10")
    file.append("origin_drop/randomDrop_10%_output_original_FL_C_1.0_LR_0.0025_CLI_10_CLI_EPOCHS_5_TOTAL_EPOCHS_30_BATCH_10")
    file.append("origin_drop/randomDrop_20%_output_original_FL_C_1.0_LR_0.0025_CLI_10_CLI_EPOCHS_5_TOTAL_EPOCHS_30_BATCH_10")
    file.append("origin_drop/randomDrop_50%_output_original_FL_C_1.0_LR_0.0025_CLI_10_CLI_EPOCHS_5_TOTAL_EPOCHS_30_BATCH_10")

    # file.append("pso_drop/best_score_70_output_PSO_FL_LR_0.0025_CLI_5_CLI_EPOCHS_5_TOTAL_EPOCHS_30_BATCH_10")
    # file.append("pso_drop/6918_randomDrop_10%_output_PSO_FL_LR_0.0025_CLI_10_CLI_EPOCHS_5_TOTAL_EPOCHS_30_BATCH_10")
    # file.append("pso_drop/6841randomDrop_20%_output_PSO_FL_LR_00025_CLI_10_CLI_EPOCHS_5_TOTAL_EPOCHS_30_BATCH_10")
    # file.append("pso_drop/6547randomDrop_50%_output_PSO_FL_LR_0.0025_CLI_10_CLI_EPOCHS_5_TOTAL_EPOCHS_30_BATCH_10")
    
    # metrics = read_csv(file[0])
    # draw(metrics, "FedPSO", "darkorange", "-", mode="acc")

    label = ["0%", "10%", "20%", "50%"]
    mark = ["-", "--", "-.", ":"]
    for i in range(4):
        metrics = read_csv(file[i])
        draw(metrics, label[i], "dimgray", mark[i], mode="failure")
    

    plt.savefig("failure.png")
    plt.clf()


def cost():
    draw(None, "Communication Cost", None, None, mode="data_size")
    plt.savefig("data.png")
    plt.clf()

if __name__ == "__main__":
    acc_graph()
    # cost()
    # failure()