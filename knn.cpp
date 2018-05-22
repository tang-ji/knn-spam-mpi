#include "knn.h"
#include "mpi.h"

#include <cstdio>
#include <limits>

using namespace std;

int Knn::Classify(Mail* mail_to_classify) const {
    int* labels = new int[k_];
    int* distances = new int[k_];
    this->InitNearestNeighbours(labels, distances);

    this->NearestNeighbours(mail_to_classify, labels, distances);

    int label = PredictedLabel(labels);

    // Frees the allocated memory
    delete[] labels;
    delete[] distances;

    return label;
}

void Knn::Classify(Mail* mail_to_classify, double* thresholds, int* predicted_labels, int num_predictions) const {
    int* labels = new int[k_];
    int* distances = new int[k_];
    this->InitNearestNeighbours(labels, distances);

    this->NearestNeighbours(mail_to_classify, labels, distances);

    for (int i = 0; i < num_predictions; i++) {
        predicted_labels[i] = PredictedLabel(labels, thresholds[i]);
    }

    // Frees the allocated memory
    delete[] labels;
    delete[] distances;
}

void Knn::InitNearestNeighbours(int* labels, int* distances) const {
    // Arrays to store the k nearest neighbours
    for (int i = 0; i < k_; ++i) {
        labels[i] = -1;
        distances[i] = std::numeric_limits<int>::max();
    }
}




void Knn::Merge(int* local_labels, int* local_distances,
        const int* other_labels, const int* other_distances) const{
  int cur_local = k_;
    int cur_other = k_;
    // Gets rid of the neighbours which are too far away
    while (cur_local + cur_other > k_) {
        if (local_distances[cur_local-1] > other_distances[cur_other-1]) {
            cur_local--;
        } else {
            cur_other--;
        }
    }
    // Merges the two arrays in the local one
    while (cur_local || cur_other ) {
        if (!cur_local || (cur_other &&
                    local_distances[cur_local-1] <= other_distances[cur_other-1])) {
            cur_other--;
            local_distances[cur_local + cur_other] = other_distances[cur_other];
            local_labels[cur_local + cur_other] = other_labels[cur_other];
        } else {
            cur_local--;
            local_distances[cur_local + cur_other] = local_distances[cur_local];
            local_labels[cur_local + cur_other] = local_labels[cur_local];
        }
    }
}


int Knn::PredictedLabel(int* labels) const {
    int num_spam = 0;
    int num_ham = 0;
    for (int i = 0; i < k_; ++i) {
        switch(labels[i]) {
            case 0:
                num_ham++;
                break;
            case 1:
                num_spam++;
                break;
            default:
                break;
        }
    }
    return num_spam > num_ham ? 1 : 0;
}

void Knn::NearestNeighbours(Mail* mail_to_classify, int* labels, int* distances) const {
	std::vector<Mail*> train_mail = train_instances_->instances();
	for(int i = 0; i < train_mail.size(); i++) {
		Insert(labels, distances, mail_to_classify, train_mail[i]);
	}
}

int Knn::ClassifyMpi(Mail* mail_to_classify) const {
	int numtasks,rank;
	MPI_Status status;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

	int* labels = new int[k_];
	int* distances = new int[k_];
	int* other_labels = new int[k_];
	int* other_distances = new int[k_];

	this->InitNearestNeighbours(labels, distances);

	int n= train_instances_->num_instances();
	for(int i = rank * n / numtasks; i < (rank + 1) * n / numtasks; i++){
		Insert(labels, distances, mail_to_classify, train_instances_->instance(i));
	} 

	if(rank == 0){
	 	for(int i = 1; i < numtasks; i++){
			MPI_Recv(other_labels, k_, MPI_INT, i, i, MPI_COMM_WORLD, &status);
			MPI_Recv(other_distances, k_, MPI_INT, i, i, MPI_COMM_WORLD, &status);
			Merge(labels, distances, other_labels, other_distances);
	 	}
	}else{
	  	MPI_Send(labels, k_, MPI_INT, 0, rank, MPI_COMM_WORLD);
	  	MPI_Send(distances, k_, MPI_INT, 0, rank, MPI_COMM_WORLD);
	}

	int result = PredictedLabel(labels);
	delete[] labels;
	delete[] distances;
	delete[] other_labels;
	delete[] other_distances;

	return result;
}
int Knn::PredictedLabel(int* labels, double threshold) const {
	int sum = 0;
	for(int i = 0; i < k_; i++) {
		sum += labels[i];
	}
	if((double)sum/k_ < threshold) return 0;
	return 1;
}

void Knn::Insert(int* labels, int* distances, Mail* mail_to_classify,
        Mail* train_mail) const {
	int distance = mail_to_classify->Distance(train_mail);
	int label = train_mail->label();
	int n = k_, i;
	if(distances[n - 1] <= distance) return;
	for(i = n - 1; i >= 0; i--) {
		if(distances[i] <= distance) break;
		if(i == n - 1) continue;	
		labels[i+1] = labels[i];	
		distances[i+1] = distances[i];
	}
	labels[i+1] = label;
	distances[i+1] = distance;
}

