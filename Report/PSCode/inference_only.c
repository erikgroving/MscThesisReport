#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdint.h>
#include "parse_mnist.h"
#include <unistd.h>
#include <math.h> 
#include <sys/time.h>
#include <time.h>
#include <string.h>

#define FORWARD   1
#define WAITING   2
#define BACKWARD  3
#define UPDATE    4
#define IDLE    5
#define SET_SIZE  70000
#define TRAIN_SIZE  70000

typedef struct ddr_data {
  // written to by fpga         Offset    Desc
  uint32_t  fpga_img_id;        // 0      fpga image ptr
  uint32_t  epoch;              // 1  
  uint32_t  num_correct_train;  // 2  
  uint32_t  num_correct_test;   // 3
  uint32_t  idle_cycles;        // 4  
  uint32_t  active_cycles;      // 5
  uint32_t  status;             // 6      contains status info
  
  // written to by arm
  uint32_t  start;              // 7      start looping
  uint32_t  n_epochs;           // 8      upper limit on epochs
  uint32_t  learning_rate;      // 9      # of right shifts
  uint32_t  training_mode;      // 10     train or just forward pass
  uint32_t  img_set_size;       // 11     size of dataset
  uint32_t  img_id;             // 12     arm image ptr
  uint32_t  img_label;          // 13
  uint32_t  img[196];           // 14
  int16_t   out[10];

} ddr_data_t;

void state_enc_to_str(uint32_t state, char* enc); 
void parse_mnist_data(char* filename, uint32_t** mnist_images);
void print_debug_data(volatile ddr_data_t* ddr_ptr);

int main() {
  uint32_t magic_number;
  uint32_t id, test_idx, epoch, corr_tr, corr_test;  
  uint32_t** train_images;
  uint32_t** test_images;
  uint32_t* train_labels;
  uint32_t* test_labels;  

  int handle = open("/dev/mem", O_RDWR | O_SYNC); 
  volatile ddr_data_t* ddr_ptr = mmap(NULL, 134217728, PROT_READ | PROT_WRITE, MAP_SHARED, handle, 0x40000000);
  
  
  uint32_t* ptr = (uint32_t*)ddr_ptr;   
  magic_number = ptr[400];
  printf("@@@ Checking Magic Number\n");
  if (magic_number != 0xFADEDBEE) {
    printf("@@@ Memory was read incorrectly.\n");
    return -1;
  }
  printf("@@@ Magic number: %08x\n", magic_number);
  printf("@@@ Magic number successfully read.\n");
  
  // Load MNIST images into memory
  printf("@@@ Loading MNIST images...\n");
  train_images = parse_mnist_images("data/train-images.idx3-ubyte");
  train_labels = parse_mnist_labels("data/train-labels.idx1-ubyte");
  test_images = parse_mnist_images("data/t10k-images.idx3-ubyte");
  test_labels = parse_mnist_labels("data/t10k-labels.idx1-ubyte");
  printf("@@@ Loading complete!\n");

  struct timespec sleep;
  sleep.tv_sec = 0;
  sleep.tv_nsec = 1000;

  // Start training! 
  ddr_ptr->start = 0;
  usleep(10);
  ddr_ptr->start = 1;
  ddr_ptr->n_epochs = 2;
  ddr_ptr->training_mode = 0;  
  ddr_ptr->img_set_size = SET_SIZE - 1;
  struct timeval start, end;
  gettimeofday(&start, NULL);
  do {
    id    = (ddr_ptr->fpga_img_id + 1) % SET_SIZE;
    epoch   = ddr_ptr->epoch;    
    // Print data if epoch just finished
    if ((id == 0) && epoch != 0) {    
      gettimeofday(&end, NULL);       
      
      corr_tr     = ddr_ptr->num_correct_train;
      corr_test   = ddr_ptr->num_correct_test;
      printf("\nImages"
          ": %d/%d\nAccuracy: %f%%\n", corr_test, 70000, 
          ((float)corr_test/70000.) * 100.);
   
           
      uint32_t active = ddr_ptr->active_cycles;
      uint32_t idle = ddr_ptr->idle_cycles;
      printf("Active Cycles: %d\t Idle Cycles: %d\n", active, idle);
      printf("Active Cycle Percentage: %f%%\n", (float)active / ((float)idle + (float)active));    
      printf("Elapsed time: %.5f seconds\n\n", (end.tv_sec - start.tv_sec) + ((end.tv_usec - start.tv_usec) * 1e-6));
      gettimeofday(&start, NULL);
    }
       
    if (id < 60000) {      
      memcpy((void*)ddr_ptr->img, train_images[id], sizeof(uint32_t) * 196);
      ddr_ptr->img_label  = train_labels[id];
    }
    else {
      test_idx = id - 60000;     
      memcpy((void*)ddr_ptr->img, test_images[test_idx],  sizeof(uint32_t) * 196);
      ddr_ptr->img_label  = test_labels[test_idx];
    }    
    
    nanosleep(&sleep, NULL);
    ddr_ptr->img_id   = id;      

  } while (epoch < ddr_ptr->n_epochs);
}
