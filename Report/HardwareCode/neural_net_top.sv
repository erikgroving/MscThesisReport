`timescale 1ns / 1ps
`include "sys_defs.vh"

module neural_net_top(
  inout                   [14:0]DDR_addr,
  inout                   [2:0]DDR_ba,
  inout                   DDR_cas_n,
  inout                   DDR_ck_n,
  inout                   DDR_ck_p,
  inout                   DDR_cke,
  inout                   DDR_cs_n,
  inout                   [3:0]DDR_dm,
  inout                   [31:0]DDR_dq,
  inout                   [3:0]DDR_dqs_n,
  inout                   [3:0]DDR_dqs_p,
  inout                   DDR_odt,
  inout                   DDR_ras_n,
  inout                   DDR_reset_n,
  inout                   DDR_we_n,
  inout                   FIXED_IO_ddr_vrn,
  inout                   FIXED_IO_ddr_vrp,
  inout                   [53:0]FIXED_IO_mio,
  inout                   FIXED_IO_ps_clk,
  inout                   FIXED_IO_ps_porb,
  inout                   FIXED_IO_ps_srstb,

  input                   rst,
  input           [7: 0]  sw_in,
  input                   clock_in,
  output  logic   [7: 0]  led_o
  );
  
  logic                                       fab_clk;
  logic                                       clk;
  logic                                       forward;
  // Logics for the fc0 layer
  logic                                       fc0_start;
  logic [1: 0][`PREC - 1: 0]                  fc0_activation_i;
  logic                                       fc0_valid;
  logic                                       fc0_valid_i;    
  logic [`FC0_NEURONS - 1: 0][`PREC - 1: 0]   fc0_activation_o ;
  logic [`FC0_NEURONS - 1: 0][6: 0]           fc0_neuron_id_o ;
  logic                                       fc0_valid_act_o; 
  logic                                       fc0_busy;    
  logic [`FC0_NEURONS - 1: 0][`PREC - 1: 0]   fc0_gradients;
  logic                                       fc0_grad_valid; 

  
  // Logics for the fc1 layer    
  logic                                       fc1_start;
  logic [`FC1_N_KERNELS - 1: 0][`PREC - 1: 0] fc1_activation_i ;
  logic                                       fc1_valid_i;    
  logic [`FC1_N_KERNELS - 1: 0][`PREC - 1: 0] fc1_activation_o;
  logic [`FC1_N_KERNELS - 1: 0][5: 0]         fc1_neuron_id_o ;
  logic                                       fc1_valid_act_o;
  logic                                       fc1_buff_rdy; 
  logic                                       fc1_busy;   
  logic                                       fc1_grad_valid; 

   // Logics for the fc2 layer (the last fc layer)
  logic                                       fc2_start;
  logic                                       fc2_buff_rdy;
  logic [`FC2_N_KERNELS - 1: 0][`PREC - 1: 0] fc2_activation_i;
  logic                                       fc2_valid_i;
  logic                                       fc2_busy;

  logic [`FC2_N_KERNELS - 1: 0][`PREC - 1: 0] fc2_activation_o;
  logic [`FC2_N_KERNELS - 1: 0][3: 0]         fc2_neuron_id_o;
  logic                                       fc2_valid_o;       
  logic [`FC2_NEURONS - 1: 0][`PREC - 1: 0]   fc2_act_o_buf;
  logic                                       fc2_buf_valid; 
  
  // Backward pass logics
  logic [`FC0_N_KERNELS - 1: 0][`PREC - 1: 0] fc0_b_gradient_i;
  logic [`FC0_N_KERNELS - 1: 0][`PREC - 1: 0] fc0_b_activation_i;
  logic [9: 0]                                fc0_b_activation_id_i;
  logic [9: 0]                                fc0_b_activation_id_o;
  logic                                       fc0_b_valid_i;
  logic                                       fc0_b_start;
  logic                                       fc0_b_start_r;
  logic [3: 0]                                fc0_loops;
  logic [`FC0_NEURONS - 1: 0][`PREC - 1: 0]   fc0_gradients_i;
  logic                                       fc0_gradients_rdy;
  logic [6: 0]                                fc0_n_loop_offset;
  logic                                       fc0_bp_done;
  logic                                       fc0_update;
  logic                                       fc0_update_done;
        

  
  logic [`FC1_N_KERNELS - 1: 0][`PREC - 1: 0] fc1_b_gradient_i;
  logic [`PREC - 1: 0]                        fc1_b_activation_i;
  logic [6: 0]                                fc1_b_activation_id_i;
  logic [6: 0]                                fc1_b_activation_id_o;
  logic [`FC1_N_KERNELS - 1: 0][5: 0]         fc1_b_neuron_id_i;
  logic                                       fc1_b_valid_i;
  logic                                       fc1_b_start;
  logic                                       fc1_b_start_r;
  logic [3: 0]                                fc1_loops;
  logic [`FC1_NEURONS - 1: 0][`PREC - 1: 0]   fc1_gradients;
  logic [`FC1_N_KERNELS - 1: 0][`PREC - 1: 0] fc1_gradients_i;
  logic                                       fc1_gradients_rdy;
  logic [5: 0]                                fc1_n_offset;
  logic [5: 0]                                fc1_n_loop_offset;
  logic                                       fc1_bp_mode;   
  logic                                       fc1_bp_done;    
  logic                                       fc1_update;
  logic                                       fc1_update_done;
        

  
  logic [`FC2_N_KERNELS - 1: 0][`PREC - 1: 0] fc2_b_gradient_i;
  logic [`PREC - 1: 0]                        fc2_b_activation_i;
  logic [5: 0]                                fc2_b_activation_id_i;
  logic [5: 0]                                fc2_b_activation_id_o;
  logic [`FC2_N_KERNELS - 1: 0][3: 0]         fc2_b_neuron_id_i;
  logic                                       fc2_b_valid_i;
  logic                                       fc2_b_start;
  logic                                       fc2_b_start_r;
  logic [3: 0]                                fc2_loops;
  logic [`FC2_NEURONS - 1: 0][`PREC - 1: 0]   fc2_gradients;
  logic [`FC2_N_KERNELS - 1: 0][`PREC - 1: 0] fc2_gradients_i;
  logic                                       fc2_gradients_rdy;
  logic [3: 0]                                fc2_n_offset;
  logic                                       fc2_bp_mode; 
  logic                                       fc2_bp_done;
  logic                                       fc2_update;
  logic                                       fc2_update_done;
  
  logic [7: 0]                                img1_unpacked[784];
  logic                                       new_img; 
  logic [9:0]                                 epoch;
  logic [16:0]                                img_id;
  logic [4: 0]                                lrate_shifts;
  logic [4: 0]                                lrate_shifts_bus;
  logic [31: 0]                               active_cycles;
  logic [31: 0]                               idle_cycles;
  logic                                       training_mode;
  logic                                       training_mode_bus;
  logic [16:0]                                img_set_size;

  logic [16:0]                                img1_id;     
  logic [16: 0]                               prev_img_id;
  logic [9:0]                                 img1_label;
  logic [9:0]                                 n_epochs;
  logic [16:0]                                num_correct_test;
  logic [16:0]                                num_correct_train;
  logic                                       start;
  logic                                       start_bus;
  
  // Layer States
  logic [2: 0]                                fc0_state;
  logic [2: 0]                                next_fc0_state;
  logic [2: 0]                                fc1_state;
  logic [2: 0]                                next_fc1_state;
  logic [2: 0]                                fc2_state;
  logic [2: 0]                                next_fc2_state;
  logic all_idle;
  
      
  logic [9: 0]                                input_addr;
  logic [`PREC - 1: 0]                        net_input_bram_dout_a;
  logic [`PREC - 1: 0]                        net_input_bram_dout_b;
  logic [`PREC - 1: 0]                        input_data_a;
  logic [`PREC - 1: 0]                        input_data_b;
  logic [9: 0]                                img_label;
  logic                                       img_rdy;  
  logic                                       epoch_fin;
  logic                                       correct;
  logic [12: 0]                               fc0_ptr_a;
  logic [12: 0]                               fc0_ptr_b;
  logic [9: 0]                                fc0_addr_a;
  logic [9: 0]                                fc0_addr_b;   
  logic [`FC2_NEURONS - 1: 0][`PREC - 1: 0]   fc2_out;
  logic [4: 0][`PREC - 1: 0]                  max1;
  logic [2: 0][`PREC - 1: 0]                  max2;
  logic [1: 0][`PREC - 1: 0]                  max3;    
  logic [`PREC - 1: 0]                        max4;   
  logic [`PREC - 1: 0]                        max;
  logic [4: 0]                                max_valid;
  logic [7: 0]                                led_o_r;
  logic                                       sm_valid_o;
  logic [`FC2_NEURONS - 1: 0][`PREC - 1: 0]   sm_grad_o;     
  logic [31: 0]                               status_block;
  
   
  localparam sf   = 2.0**-12.0;
  localparam sf2  = 2.0**-17.0;
  
  // Backward pass states
  localparam WEIGHT_MODE = 0;
  localparam NEURON_MODE = 1;
  
  // Layer states
  localparam FORWARD  = 1;
  localparam WAITING  = 2;
  localparam BACKWARD = 3;
  localparam UPDATE   = 4;
  localparam IDLE   = 5;

  
  mmcm_50_mhz mmcm_50_mhz_i (
    .clk_in1(fab_clk),
    //.clk_in1(clock_in),
    .clk_out1(clk)
  );
  
  logic [7: 0] sw_i; // so simulation uses net_input_bram   
  assign sw_i = sw_in;

  
  assign start         = sw_i[0] ? 1'b1 : start_bus;
  assign training_mode = sw_i[0] ? 1'b1 : training_mode_bus;
  assign forward       = fc0_state == FORWARD || fc1_state == FORWARD || fc2_state == FORWARD;
  assign all_idle      = (fc0_state == IDLE) && (fc1_state == IDLE) && (fc2_state == IDLE);
  assign img_rdy       = (img1_id == (img_id + 1'b1)) | (img1_id == 0 && img_id == img_set_size);
  assign new_img       = start & all_idle & img_rdy;
  assign epoch_fin     = sw_i[0] ? 1'b0 : epoch == n_epochs;
  
  logic reset_i;
  logic reset;
  always_ff @(posedge clk) begin
    reset_i       <= rst;
    lrate_shifts  <= sw_i[0] ? 5'd7 : lrate_shifts_bus;
  end
  
  always_ff @(posedge clk) begin
    if (reset || !start) begin
      idle_cycles     <= 0;
      active_cycles   <= 0;      
    end
    else begin
      idle_cycles     <= idle_cycles + all_idle;
      active_cycles   <= all_idle ? active_cycles : active_cycles + 1'b1;
    end
  end
  
  BUFG BUFG_reset(.I(reset_i), .O(reset));
  
  always_ff @(posedge clk) begin
    if (reset) begin
      input_addr  <= 0;
      fc0_start   <= 0;
    end
    else if (fc0_state == FORWARD & !fc0_start && ~epoch_fin) begin
      fc0_start   <= 1'b1;
      input_addr  <= 0;
    end
    else if (fc0_state == FORWARD & fc0_start) begin
      input_addr  <= input_addr + 1'b1;
    end
    else begin
      fc0_start   <= 1'b0;
      input_addr  <= 0;
    end
  end
  
  


  assign fc0_addr_a = (forward) ? input_addr << 1 : fc0_b_activation_id_i << 1; 
  assign fc0_addr_b = fc0_addr_a + 1'b1;
  

  net_input_bram net_input_bram_i (
    .addra(fc0_addr_a),
    .clka(clk),
    .dina(18'b0),
    .douta(net_input_bram_dout_a),
    .ena(1'b1),
    .wea(1'b0),
    
    .addrb(fc0_addr_b),
    .clkb(clk),
    .dinb(18'b0),
    .doutb(net_input_bram_dout_b),
    .enb(1'b1),
    .web(1'b0)
  );  

  
    

  always_ff @(posedge clk) begin
    if (reset) begin
      prev_img_id <= img_set_size;
    end
    else begin
      prev_img_id <= img_id;
    end

        
    if (reset || (img_id == 0 && prev_img_id != 0)) begin
      num_correct_train <= 0;
      num_correct_test  <= 0;
    end
    else if (correct) begin
      num_correct_train <= (~training_mode) ?
                  num_correct_train : num_correct_train + 1'b1;
      num_correct_test  <= (training_mode) ? 
                  num_correct_test : num_correct_test + 1'b1;
    end
    
    if (reset) begin
      epoch   <= 0;
    end
    else if (img_id == 0 && prev_img_id != 0) begin
      epoch   <= epoch + 1'b1; 
    end
  end
  
  
  always_comb begin
    input_data_a  <= sw_i[0] ? net_input_bram_dout_a  : 
              {6'b0, img1_unpacked[fc0_addr_a], 4'b0};
    input_data_b  <= sw_i[0] ? net_input_bram_dout_b : 
              {6'b0, img1_unpacked[fc0_addr_b], 4'b0};
  end

   
  always_ff @(posedge clk) begin
    if (reset) begin
      fc0_valid         <= 0;
      fc0_valid_i       <= 0;
    end
    else begin
      fc0_valid         <= fc0_start;
      fc0_valid_i       <= fc0_valid;      
      fc0_activation_i  <= {input_data_b, input_data_a};
    end
  end
  
  
  assign fc0_b_activation_i   = {{`FC0_NEURONS{input_data_b}}, {`FC0_NEURONS{input_data_a}}};
  assign fc0_gradients_rdy  = fc0_grad_valid; 
  // Start when backward is good and gradients are ready. Only do backprop once   
  assign fc0_b_start = fc0_state == BACKWARD;
  bit [7: 0] q, r;
  always_ff @(posedge clk) begin
    if (reset) begin
      fc0_b_start_r   <= 1'b0;
    end
    else begin
      fc0_b_start_r   <= fc0_b_start;
    end
    
    // Loop over fan in
    if (reset) begin
      fc0_loops   <= 0;
    end
    else if (fc0_state != BACKWARD) begin
      fc0_loops   <= 0;
    end
    else if (fc0_b_activation_id_i == (`FC0_KERNEL_FAN_IN - 1)) begin
      fc0_loops   <= fc0_loops + 1'b1;
    end
    
    if (reset) begin
      fc0_b_activation_id_i <= 0;     
    end
    else if (fc0_state != BACKWARD) begin
      fc0_b_activation_id_i <= 0;
    end
    else if (fc0_b_start) begin
      fc0_b_activation_id_i <= (fc0_b_activation_id_i == (`FC0_KERNEL_FAN_IN - 1'b1)) ? 
                    0 : fc0_b_activation_id_i + 1'b1;
    end
    
    for (q = 0, r = `FC0_PORT_WIDTH; q < `FC0_PORT_WIDTH; q=q+1, r=r+1) begin
      fc0_gradients_i[q]    <= fc0_gradients[q];
    end
    fc0_b_activation_id_o     <= fc0_b_activation_id_i << 1;
  end
  always_comb begin
    case(fc0_state)
      FORWARD:
        next_fc0_state  = fc1_buff_rdy & training_mode 
                            ? WAITING   :  
                  fc1_buff_rdy & ~training_mode
                            ? IDLE    : FORWARD;                   
      WAITING:
        next_fc0_state  = (fc0_gradients_rdy)   ? BACKWARD  : WAITING;        
      BACKWARD:
        next_fc0_state  = (fc0_bp_done)     ? UPDATE  : BACKWARD;
      UPDATE:
        next_fc0_state  = (fc0_update_done)   ? IDLE    : UPDATE;
      IDLE:
        next_fc0_state  = (new_img | sw_i[0])   ? FORWARD   : IDLE ;
      default:
        next_fc0_state  = IDLE;
    endcase   
  end
  always_ff @(posedge clk) begin
    if (reset) begin
      fc0_state   <= IDLE;
    end
    else begin
      fc0_state   <= next_fc0_state;       
    end
  end
  
  assign fc0_update = fc0_state == UPDATE;
  // FC0  
  fc0_layer fc0_layer_i (
    // inputs
    .clk(clk),
    .rst(reset),  
    .forward(forward),
    .update(fc0_update),
    .activations_i(fc0_activation_i),
    .valid_i(fc0_valid_i & forward),
    .lrate_shifts(lrate_shifts),
     
    // backward pass inputs
    .b_gradient_i(fc0_gradients_i),
    .b_activation_i(fc0_b_activation_i),
    .b_activation_id(fc0_b_activation_id_o),
    .b_valid_i(fc0_b_start_r),
     
    // outputs
    .activation_o(fc0_activation_o),
    .neuron_id_o(fc0_neuron_id_o),
    .valid_act_o(fc0_valid_act_o),
    .fc0_busy(fc0_busy),
    .bp_done(fc0_bp_done),
    .update_done(fc0_update_done)
  );
  
  always_ff @(posedge clk) begin
    if (reset) begin
      fc1_start   <= 1'b0;
    end
    else begin
      fc1_start   <= fc1_state == FORWARD & fc1_buff_rdy;
    end
  end 
  interlayer_activation_buffer
  #(.N_KERNELS_I(`FC0_NEURONS), 
    .N_KERNELS_O(`FC1_N_KERNELS), 
    .ID_WIDTH(7), 
    .BUFF_SIZE(`FC0_NEURONS),
    .LOOPS(4)) 
  interlayer_activations_fc0_fc1 (
    // inputs
    .clk(clk),
    .rst(reset),
    
    .start(fc1_start),
    .activation_i(fc0_activation_o),
    .neuron_id_i(fc0_neuron_id_o),
    .valid_act_i(fc0_valid_act_o & forward),
    .b_ptr(fc1_b_activation_id_i),
    // outputs
    .activation_o(fc1_activation_i),
    .valid_o(fc1_valid_i),
    
    .b_act_o(fc1_b_activation_i),
    
    .buff_rdy(fc1_buff_rdy)
  );
  
  
  
  assign fc1_gradients_rdy  = fc1_grad_valid;  
  assign fc1_n_offset       = (fc1_loops >= `FC1_MODE_SWITCH) ? fc1_loops - 4 : fc1_loops;
  // Start when backward is good and gradients are ready. Only do backprop once   
  assign fc1_b_start        = fc1_state == BACKWARD;
  bit [5: 0] o, p;
  always_ff @(posedge clk) begin
    if (reset) begin
      fc1_b_start_r   <= 1'b0;
      fc1_bp_mode   <= 1'b0;
    end
    else begin
      fc1_b_start_r   <= fc1_b_start;
      fc1_bp_mode   <= fc1_loops >= `FC1_MODE_SWITCH ? WEIGHT_MODE : NEURON_MODE;
    end
    
    // Loop over fan in
    if (reset) begin
      fc1_loops   <= 0;
    end
    else if (fc1_state != BACKWARD) begin
      fc1_loops   <= 0;
    end
    else if (fc1_b_activation_id_i == (`FC0_NEURONS - 1)) begin
      fc1_loops   <= fc1_loops + 1'b1;
    end
    
    if (reset) begin
      fc1_b_activation_id_i <= 0;     
    end
    else if (fc1_state != BACKWARD) begin
      fc1_b_activation_id_i <= 0;
    end
    else if (fc1_b_start) begin
      fc1_b_activation_id_i <= (fc1_b_activation_id_i == (`FC1_FAN_IN - 1'b1)) ? 
                    0 : fc1_b_activation_id_i + 1'b1;
    end
    
    for (p = 0, o = `FC1_PORT_WIDTH; p < `FC1_PORT_WIDTH; p=p+1, o=o+1) begin
      fc1_gradients_i[p]    <= fc1_gradients[(fc1_n_offset << 3) + p];
      fc1_gradients_i[o]    <= fc1_gradients[((fc1_n_offset << 3) + p) | 6'd32];
      fc1_b_neuron_id_i[p]  <= (fc1_n_offset << 3) + p;    
      fc1_b_neuron_id_i[o]  <= ((fc1_n_offset << 3) + p) | 6'd32;
    end
    fc1_b_activation_id_o   <= fc1_b_activation_id_i;
  end

  always_comb begin
    case(fc1_state)
      FORWARD:
        next_fc1_state  = fc2_buff_rdy & training_mode 
                            ? WAITING   :  
                  fc2_buff_rdy & ~training_mode
                            ? IDLE    : FORWARD;                     
      WAITING:
        next_fc1_state  = (fc1_gradients_rdy)   ? BACKWARD  : WAITING;        
      BACKWARD:
        next_fc1_state  = (fc1_bp_done)     ? UPDATE  : BACKWARD;
      UPDATE:
        next_fc1_state  = (fc1_update_done)   ? IDLE    : UPDATE;
      IDLE:
        next_fc1_state  = (new_img | sw_i[0])   ? FORWARD   : IDLE ;
      default:
        next_fc1_state  = IDLE;
    endcase   
  end
  always_ff @(posedge clk) begin
    if (reset) begin
      fc1_state   <= IDLE;
    end
    else begin
      fc1_state   <= next_fc1_state;       
    end
  end
  
  assign fc1_update = fc1_state == UPDATE;
  // FC1   
  fc1_layer fc1_layer_i (
    // inputs
    .clk(clk),
    .rst(reset),  
    .forward(forward), 
    .update(fc1_update),
    .activations_i(fc1_activation_i),
    .valid_i(fc1_valid_i & forward),    
    .lrate_shifts(lrate_shifts),
    
    // backward pass inputs
    .b_gradient_i(fc1_gradients_i),
    .b_activation_i({`FC1_N_KERNELS{fc1_b_activation_i}}),
    .b_activation_id(fc1_b_activation_id_o),
    .b_neuron_id_i(fc1_b_neuron_id_i),
    .b_valid_i(fc1_b_start_r),
    .bp_mode(fc1_bp_mode),

    // outputs
    .activation_o(fc1_activation_o),
    .neuron_id_o(fc1_neuron_id_o),
    .valid_act_o(fc1_valid_act_o),
    .fc1_busy(fc1_busy), 
    .bp_done(fc1_bp_done),
    .update_done(fc1_update_done),
    
    // backward pass outputs
    .pl_gradients(fc0_gradients),
    .pl_grad_valid(fc0_grad_valid) 
  );
  
     
  always_ff @(posedge clk) begin
    if (reset) begin
      fc2_start   <= 1'b0;
    end
    else begin
      fc2_start   <= fc2_state == FORWARD & fc2_buff_rdy;
    end
  end 
 

  interlayer_activation_buffer
  #(.N_KERNELS_I(`FC1_N_KERNELS), 
    .N_KERNELS_O(`FC2_N_KERNELS), 
    .ID_WIDTH(6), 
    .BUFF_SIZE(`FC1_NEURONS),
    .LOOPS(`FC2_NEURONS)) 
  interlayer_activations_fc1_fc2 (
    // inputs
    .clk(clk),
    .rst(reset),
    
    .start(fc2_start),
    .activation_i(fc1_activation_o),
    .neuron_id_i(fc1_neuron_id_o),
    .valid_act_i(fc1_valid_act_o & forward),
    .b_ptr(fc2_b_activation_id_i),
    // outputs
    
    .activation_o(fc2_activation_i),
    .valid_o(fc2_valid_i),
    
    .b_act_o(fc2_b_activation_i),
    
    .buff_rdy(fc2_buff_rdy)
  );
  
  always_comb begin
    case(fc2_state)
      FORWARD:
        next_fc2_state  =     fc2_buf_valid & training_mode 
                            ? WAITING   :  
                              fc2_buf_valid & ~training_mode
                            ? IDLE    : FORWARD;               
      WAITING:
        next_fc2_state  = (fc2_gradients_rdy) ? BACKWARD  : WAITING;        
      BACKWARD:
        next_fc2_state  = (fc2_bp_done)       ? UPDATE  : BACKWARD;
      UPDATE:
        next_fc2_state  = (fc2_update_done)   ? IDLE    : UPDATE;
      IDLE:
        next_fc2_state  = (new_img | sw_i[0]) ? FORWARD   : IDLE;
      default:
        next_fc2_state  = IDLE;
    endcase   
  end
  always_ff @(posedge clk) begin
    if (reset) begin
      fc2_state   <= IDLE;
    end
    else begin
      fc2_state   <= next_fc2_state;       
    end
  end  

  

  
  assign fc2_n_offset = (fc2_loops >= `FC2_MODE_SWITCH) ? fc2_loops - 5 : fc2_loops;

  // Start when backward is good and gradients are ready. Only do backprop once   
  assign fc2_b_start = fc2_state == BACKWARD;
  always_ff @(posedge clk) begin
    if (reset) begin
      fc2_b_start_r   <= 1'b0;
      fc2_bp_mode   <= 1'b0;
    end
    else begin
      fc2_b_start_r   <= fc2_b_start;
      fc2_bp_mode   <= fc2_loops >= `FC2_MODE_SWITCH ? WEIGHT_MODE : NEURON_MODE;
    end
    
    // Loop over fan in
    if (reset) begin
      fc2_loops   <= 0;
    end
    else if (fc2_state != BACKWARD) begin
      fc2_loops   <= 0;
    end
    else if (fc2_b_activation_id_i == (`FC1_NEURONS - 1)) begin
      fc2_loops   <= fc2_loops + 1'b1;
    end
    
    
    if (reset) begin
      fc2_b_activation_id_i <= 0;     
    end
    else if (fc2_state != BACKWARD) begin
      fc2_b_activation_id_i <= 0;
    end
    else if (fc2_b_start) begin
      fc2_b_activation_id_i <= fc2_b_activation_id_i + 1'b1;
    end
    fc2_gradients_i       <= {fc2_gradients[fc2_n_offset + 5], fc2_gradients[fc2_n_offset]};
    fc2_b_neuron_id_i     <= {fc2_n_offset + 5, fc2_n_offset};
    fc2_b_activation_id_o <= fc2_b_activation_id_i;
  end
  
  assign fc2_update = fc2_state == UPDATE;
  // FC2, fed directly from FC1 due to the small size
  fc2_layer fc2_layer_i (
    // inputs
    .clk(clk),
    .rst(reset),
    .forward(forward),
    .update(fc2_update),
    .activations_i(fc2_activation_i),
    .valid_i(fc2_valid_i & forward),
    .lrate_shifts(lrate_shifts),
     
    // backward pass inputs
    .b_gradient_i(fc2_gradients_i),
    .b_activation_i({fc2_b_activation_i, fc2_b_activation_i}),
    .b_activation_id(fc2_b_activation_id_o),
    .b_neuron_id_i(fc2_b_neuron_id_i),
    .b_valid_i(fc2_b_start_r),
    .bp_mode(fc2_bp_mode),
  
    // outputs
    .activation_o(fc2_activation_o),
    .neuron_id_o(fc2_neuron_id_o),
    .valid_act_o(fc2_valid_o),
    .fc2_busy(fc2_busy),
    .bp_done(fc2_bp_done),
    .update_done(fc2_update_done),
    
    // backward pass outputs
    .pl_gradients(fc1_gradients),
    .pl_grad_valid(fc1_grad_valid)
  );






  bit [`FC2_N_KERNELS - 1: 0] m;
  logic prev_fc2_buf_valid;
  always_ff @(posedge clk) begin
    if (reset) begin
      prev_fc2_buf_valid  <= 0;
      fc2_act_o_buf       <= 0;
    end
    else begin
      prev_fc2_buf_valid  <= fc2_buf_valid;
      for (m = 0; m < `FC2_N_KERNELS; m=m+1) begin
        if (fc2_valid_o && forward) begin
          fc2_act_o_buf[fc2_neuron_id_o[m]]  <= fc2_activation_o[m];
        end 
      end
    end
    
    if (reset) begin
      fc2_buf_valid   <= 1'b0;
    end
    else if (fc2_valid_o) begin
      fc2_buf_valid   <= fc2_neuron_id_o[`FC2_N_KERNELS - 1] == `FC2_NEURONS - 1;
    end
    else if (fc2_state == IDLE) begin
      fc2_buf_valid   <= 1'b0;
    end
  end
  
  always @(posedge clk) begin
    if (fc2_buf_valid) begin
      fc2_out <= fc2_act_o_buf;
    end
  end
  
  
  
  
  // LED Logic  
  bit [3: 0] k;
  bit [3: 0] j, t;
  always_ff @(posedge clk) begin
    if (reset) begin
      max         <= 0;
      max_valid   <= 0;
    end
    else if ({fc2_buf_valid, prev_fc2_buf_valid} == 2'b10) begin 
      for (k = 0; k < 5; k=k+1) begin
        max1[k] <= $signed(fc2_act_o_buf[2*k]) > $signed(fc2_act_o_buf[2*k+1]) ? 
              fc2_act_o_buf[2*k] : fc2_act_o_buf[2*k + 1];
      end
      max_valid     <= {max_valid[3: 0], 1'b1};
    end
    else begin
      max_valid[0]  <= 1'b0;
      
      max2[0]       <= $signed(max1[0]) > $signed(max1[1]) ? max1[0] : max1[1];
      max2[1]       <= $signed(max1[2]) > $signed(max1[3]) ? max1[2] : max1[3];
      max2[2]       <= max1[4];
      max_valid[1]  <= max_valid[0];
      
      max3[0]       <= $signed(max2[0]) > $signed(max2[1]) ? max2[0] : max2[1];
      max3[1]       <= max2[2];
      max_valid[2]  <= max_valid[1];
      
      max4          <= $signed(max3[0]) > $signed(max3[1]) ? max3[0] : max3[1];
      max_valid[3]  <= max_valid[2];
      
      max           <= max4;
      max_valid[4]  <= max_valid[3];   
         

    end  
    if (reset) begin
      led_o_r   <= 0;
      correct   <= 1'b0;
    end     
    else if (max_valid[4]) begin
      correct  <= fc2_act_o_buf[img_label] == max;
      for (t = 0; t < `FC2_NEURONS; t=t+1) begin
        if (fc2_act_o_buf[t] == max && t != img_label) begin
          correct <= 1'b0;
        end
      end
      for (j = 0; j < 8; j=j+1) begin
        led_o_r[j] <= fc2_act_o_buf[j] == max;
      end
    end
    else begin
      correct   <= 1'b0;
    end
    led_o[7:0]  <= led_o_r[7: 0];
  end
  

  softmax softmax_i (
    .clk(clk),
    .reset(reset),
    .start(max_valid[4]),
    .max(max),
    .act_in(fc2_act_o_buf),
    
    .valid_o(sm_valid_o),
    .grad_o(sm_grad_o)
  );
 
  bit [3: 0] u;                    
  always_ff @(posedge clk) begin
    if (reset) begin
      fc2_gradients_rdy     <= 0;
    end
    else if (all_idle) begin
      fc2_gradients_rdy     <= 1'b0;
    end
    else if (sm_valid_o) begin
      fc2_gradients_rdy     <= 1'b1;
    end
    
    if (sm_valid_o) begin
      for (u = 0; u < `FC2_NEURONS; u=u+1) begin
        fc2_gradients[u]  <= (fc2_act_o_buf[img_label] == `MIN_VAL) ? 0 :
                      sm_grad_o[u];
      end
      fc2_gradients[img_label]  <= (fc2_act_o_buf[img_label] == `MAX_VAL) ? 0 : 
                      $signed(sm_grad_o[img_label]) - $signed(`ONE);
    end
  end

   
  assign status_block = {5'b0, led_o_r, fc0_state, fc1_state, fc2_state, forward, fc0_start,
             fc1_start, fc2_start, fc0_busy, fc1_busy, fc2_busy, new_img, 
             all_idle, img_rdy};
             
  
  logic [31:0]img1_blk0_0;
  logic [31:0]img1_blk100_0;
  logic [31:0]img1_blk101_0;
...
  
  
system_wrapper system_wrapper_i
   (DDR_addr,
  DDR_ba,
  DDR_cas_n,
  DDR_ck_n,
  DDR_ck_p,
  DDR_cke,
  DDR_cs_n,
  DDR_dm,
  DDR_dq,
  DDR_dqs_n,
  DDR_dqs_p,
  DDR_odt,
  DDR_ras_n,
  DDR_reset_n,
  DDR_we_n,
  fab_clk,
  FIXED_IO_ddr_vrn,
  FIXED_IO_ddr_vrp,
  FIXED_IO_mio,
  FIXED_IO_ps_clk,
  FIXED_IO_ps_porb,
  FIXED_IO_ps_srstb,
  active_cycles,
  epoch,
  img_id,
  idle_cycles,
  img1_blk0_0,
  img1_blk100_0,
...
  img1_id,
  img1_label,
  img_set_size,
  lrate_shifts_bus,
  n_epochs,
  num_correct_test,
  num_correct_train,
  {fc2_out[1][17:2], fc2_out[0][17:2]},
  {fc2_out[3][17:2], fc2_out[2][17:2]},
  {fc2_out[5][17:2], fc2_out[4][17:2]},
  {fc2_out[7][17:2], fc2_out[6][17:2]},
  {fc2_out[9][17:2], fc2_out[8][17:2]},
  start_bus,
  status_block,
  training_mode_bus);

  always_ff @(posedge clk) begin
    if (reset) begin
      img_id      <= img_set_size;
      img_label   <= 0;
    end
    else if (new_img) begin
      img_id      <= img1_id;
      img_label   <= img1_label;
    end
    if (new_img) begin
      img1_unpacked[0]	<= img1_blk0_0[7:0];
      img1_unpacked[1]	<= img1_blk0_0[15:8];
      img1_unpacked[2]	<= img1_blk0_0[23:16];
      img1_unpacked[3]	<= img1_blk0_0[31:24];
      img1_unpacked[4]	<= img1_blk1_0[7:0];
...
    end
  end
  
endmodule
