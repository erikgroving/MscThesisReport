`timescale 1ns / 1ps
`include "sys_defs.vh"

module fc0_layer(
      input clk,
      input rst,
      input forward,
      input update,
      input  [1: 0][`PREC - 1: 0] activations_i,
      input valid_i,    
      input  [4: 0] lrate_shifts,    

      input [`FC0_NEURONS - 1: 0][`PREC - 1: 0] b_gradient_i,
      input [`FC0_N_KERNELS - 1: 0][`PREC - 1: 0] b_activation_i,
      input [9: 0] b_activation_id,
      input  b_valid_i,
      
      output logic [`FC0_NEURONS - 1: 0][`PREC - 1: 0] activation_o,
      output logic [`FC0_NEURONS - 1: 0][6: 0] neuron_id_o,
      output logic valid_act_o,
      output logic fc0_busy,
      output logic bp_done,
      output logic update_done
  );
  
  logic   [`FC0_PORT_WIDTH - 1: 0][`PREC - 1: 0]  data_in_a;
  logic   [`FC0_PORT_WIDTH - 1: 0][`PREC - 1: 0]  data_in_b;
  logic   [`FC0_PORT_WIDTH - 1: 0][`PREC - 1: 0]  data_out_a;
  logic   [`FC0_PORT_WIDTH - 1: 0][`PREC - 1: 0]  data_out_b;

  logic   [`FC0_N_KERNELS - 1: 0][`PREC - 1: 0]   weights;
  logic   [`FC0_ADDR - 1: 0]                      head_ptr;
  logic   [`FC0_ADDR - 1: 0]                      mid_ptr;    
  logic   [`FC0_ADDR - 1: 0]                      addr_a;
  logic   [`FC0_ADDR - 1: 0]                      addr_b;
  logic   [`FC0_BIAS_ADDR - 1: 0]                 bias_ptr;
  
  logic   [1: 0][`PREC - 1: 0]                    sch_activations;
  logic                                           sch_valid;
  logic   [1: 0][`PREC - 1: 0]                    bram_activations;
  logic                                           bram_valid;    
  logic   [`FC0_N_KERNELS - 1: 0][`PREC - 1: 0]   kern_activations;
  logic                                           kern_valid;
  
  logic   [`FC0_N_KERNELS - 1: 0][`PREC - 1: 0]   bias;
  logic   [`FC0_N_KERNELS - 1: 0][`PREC - 1: 0]   kern_bias;
  logic   [255: 0]                                bias_container;
  logic                                           sch_has_bias;
  logic                                           bram_has_bias;
  logic                                           kern_has_bias;
  logic   [`FC0_NEURONS - 1: 0][6: 0]             neuron_id;
  logic   [`FC0_N_KERNELS - 1: 0][6: 0]           kern_neuron_id;
  logic   [`FC0_N_KERNELS - 1: 0]                 last_weight;

  logic   [`FC0_N_KERNELS - 1: 0]                 valid;
  logic   [`FC0_N_KERNELS - 1: 0][`PREC - 1: 0]   kern_activation_o;
  logic   [`FC0_N_KERNELS - 1: 0][`PREC - 1: 0]   activation_o_rel;
  logic   [`FC0_N_KERNELS - 1: 0][6: 0]           kern_neuron_id_o;   
  
  
  logic [`FC0_N_KERNELS - 1: 0][`PREC - 1: 0]     b_gradient;
  logic [`FC0_N_KERNELS - 1: 0][`PREC - 1: 0]     b_gradient_pl;
  logic [`FC0_N_KERNELS - 1: 0][`PREC - 1: 0]     b_kern_grad;
  logic [`FC0_N_KERNELS - 1: 0][`PREC - 1: 0]     b_act;   
  logic [`FC0_N_KERNELS - 1: 0][`PREC - 1: 0]     b_act_pl;   
  logic [`FC0_N_KERNELS - 1: 0][`PREC - 1: 0]     b_kern_act;   
  
  logic [`FC0_N_KERNELS - 1: 0][`PREC - 1: 0]     b_kern_grad_o;
  logic [`FC0_N_KERNELS - 1: 0]                   b_kern_valid_o;
  logic [2: 0]                                    b_valid;
  logic [3: 0][9: 0]                              b_act_id;
  
  logic                                           b_weight_we;
  
  logic [`FC0_N_KERNELS - 1: 0][`PREC - 1: 0]     kern_mult1;
  logic [`FC0_N_KERNELS - 1: 0][`PREC - 1: 0]     kern_mult2;   
  logic [`FC0_N_KERNELS - 1: 0][`PREC - 1: 0]     weight_grad_o;
  logic [`FC0_N_KERNELS - 1: 0][`PREC - 1: 0]     weight_grad;
  logic [1: 0][9: 0]                              fc0_weight_grad_addr;    
  logic [1: 0][9: 0]                              fc0_weight_grad_addr_offset;
  logic [`FC0_NEURONS - 1: 0]                     act_o_sign;
  logic [`FC0_N_KERNELS - 1: 0][`PREC: 0]         update_weights_sat;
  logic [`FC0_N_KERNELS - 1: 0][`PREC - 1: 0]     update_weights;
  logic [10: 0]                                   update_ptr; 
  logic [9: 0]                                    update_addr_a;
  logic [9: 0]                                    update_addr_b;
  logic [9: 0]                                    w_addr_a;
  logic [9: 0]                                    w_addr_b;
  logic [9: 0]                                    wg_addr_a;
  logic [9: 0]                                    wg_addr_b;
  logic                                           w_we;
  logic                                           wg_we;
  
  logic                                           sch_valid_i; 
  localparam WEIGHT_MODE = 0;
  localparam NEURON_MODE = 1;   
  logic bp_mode;
  assign bp_mode = WEIGHT_MODE;  
  always_ff @(posedge clk) begin
    if (rst) begin
      sch_valid     <= 0;
    end
    else begin
      sch_valid     <= valid_i;
    end
    sch_activations <= activations_i;
  end
  
  assign sch_valid_i = (forward) ? valid_i : b_valid_i;

  // Scheduler for the fully connected layer
  fc_scheduler #(.ADDR(`FC0_ADDR), .BIAS_ADDR(`FC0_BIAS_ADDR), 
  .MID_PTR_OFFSET(`FC0_KERNEL_FAN_IN), .FAN_IN(`FC0_FAN_IN)) 
  fc0_scheduler_i (
    //inputs
    .clk(clk),
    .rst(rst),
    .forward(forward),
    .valid_i(sch_valid_i),
    
    //outputs
    .head_ptr(head_ptr),
    .mid_ptr(mid_ptr),
    .bias_ptr(bias_ptr),
    .has_bias(sch_has_bias)
  );

  


  always_ff @(posedge clk) begin
    if (rst) begin
      bram_valid    <= 0;
      bram_has_bias   <= 0;
      fc0_busy    <= 0;
    end
    else begin
      bram_valid    <= sch_valid;
      bram_has_bias   <= sch_has_bias;
      fc0_busy    <= valid_i;
    end
    bram_activations  <= sch_activations;
  end
  
  
  

  always_ff @(posedge clk) begin
    if (rst) begin
      update_ptr  <= 0;
    end
    else if (update) begin 
      update_ptr  <= update_ptr + 1'b1;
    end
    else begin
      update_ptr  <= 0;
    end
  end
  
  
  assign update_done    = update_ptr == 11'd783;
  assign update_addr_a  = update_ptr[10: 1] << 1;
  assign update_addr_b  = update_addr_a + 1'b1;
  assign w_addr_a       = (update) ? update_addr_a  : addr_a;
  assign w_addr_b       = (update) ? update_addr_b  : addr_b;
  assign wg_addr_a      = (update) ? update_addr_a  : fc0_weight_grad_addr[0];
  assign wg_addr_b      = (update) ? update_addr_b  : fc0_weight_grad_addr[1];
  assign w_we           = (update) ? update_ptr[0]  : 1'b0;  // write when odd
  assign wg_we          = (update) ? 1'b0       : b_weight_we; 
  assign addr_a         = (head_ptr << 1);
  assign addr_b         = (head_ptr << 1) + 1'b1;  
  bit [7: 0] a,c;
  always_comb begin
    weight_grad = 0;
    for (a = 0, c =`FC0_PORT_WIDTH; a < `FC0_PORT_WIDTH; a = a + 1, c=c+1) begin
      case(lrate_shifts)
        5'd7: begin
          weight_grad[a] = {{7{weight_grad_o[a][`PREC - 1]}}, {weight_grad_o[a][`PREC - 1: 7]}};
          weight_grad[c] = {{7{weight_grad_o[c][`PREC - 1]}}, {weight_grad_o[c][`PREC - 1: 7]}};   
        end  

        5'd9: begin
          weight_grad[a] = {{9{weight_grad_o[a][`PREC - 1]}}, {weight_grad_o[a][`PREC - 1: 9]}};
          weight_grad[c] = {{9{weight_grad_o[c][`PREC - 1]}}, {weight_grad_o[c][`PREC - 1: 9]}};   
        end  
        5'd11: begin
          weight_grad[a] = {{11{weight_grad_o[a][`PREC - 1]}}, {weight_grad_o[a][`PREC - 1: 11]}};
          weight_grad[c] = {{11{weight_grad_o[c][`PREC - 1]}}, {weight_grad_o[c][`PREC - 1: 11]}};   
        end    
        5'd10: begin
          weight_grad[a] = {{10{weight_grad_o[a][`PREC - 1]}}, {weight_grad_o[a][`PREC - 1: 10]}};
          weight_grad[c] = {{10{weight_grad_o[c][`PREC - 1]}}, {weight_grad_o[c][`PREC - 1: 10]}};   
        end    
        default: begin
          weight_grad[a] = {{8{weight_grad_o[a][`PREC - 1]}}, {weight_grad_o[a][`PREC - 1: 8]}};
          weight_grad[c] = {{8{weight_grad_o[c][`PREC - 1]}}, {weight_grad_o[c][`PREC - 1: 8]}}; 
        end  
      endcase
      update_weights_sat[a]   = $signed(data_out_a[a]) - $signed(weight_grad[a]);
      update_weights_sat[c]   = $signed(data_out_b[a]) - $signed(weight_grad[c]);
    end 
  end 
  
  bit [7: 0] d;
  always_comb begin
    for (d = 0; d < `FC0_N_KERNELS; d=d+1) begin
      if (update_weights_sat[d][`PREC:`PREC - 1] == 2'b01) begin
        update_weights[d]   = `MAX_VAL;
      end
      else if (update_weights_sat[d][`PREC:`PREC - 1] == 2'b10) begin
        update_weights[d]   = `MIN_VAL;
      end
      else begin
        update_weights[d]   = update_weights_sat[d][`PREC - 1: 0];
      end
    end
  end   
  

  // BRAM for the weights of the fully connected layer
  fc0_weight_bram_controller fc0_weight_bram_controller_i (
    // inputs
    .clk(clk),
    .rst(rst),
    
    .addr_a(w_addr_a),
    .data_in_a(update_weights[97: 0]),
    .en_a(1'b1),
    .we_a(w_we),
    
    .addr_b(w_addr_b),
    .data_in_b(update_weights[195: 98]),
    .en_b(1'b1),
    .we_b(w_we),
    
    // outputs
    .data_out_a(data_out_a),
    .data_out_b(data_out_b),
    .neuron_id(neuron_id)
  ); 
  
  assign b_weight_we = &b_kern_valid_o;
  
  assign fc0_weight_grad_addr_offset[0] = 0;
  assign fc0_weight_grad_addr_offset[1] = fc0_weight_grad_addr_offset[0] + 1'b1;
  assign fc0_weight_grad_addr[0]        = fc0_weight_grad_addr_offset[0] + b_act_id[3];
  assign fc0_weight_grad_addr[1]        = fc0_weight_grad_addr_offset[1] + b_act_id[3];

  assign bp_done = fc0_weight_grad_addr[1] == `FC0_FAN_IN - 1'b1;
  
  fc0_weight_gradients fc0_weight_gradients_i (
    .addra(wg_addr_a),
    .clka(clk),
    .dina(b_kern_grad_o[97: 0]),
    .douta(weight_grad_o[97: 0]),
    .ena(1'b1),
    .wea(wg_we),
    
    .addrb(wg_addr_b),
    .clkb(clk),
    .dinb(b_kern_grad_o[195: 98]),
    .doutb(weight_grad_o[195: 98]),
    .enb(1'b1),
    .web(wg_we)
  );  
  
  assign bias = 0;


  always_ff @(posedge clk) begin
    if (rst) begin
      kern_valid      <= 0;
      kern_has_bias   <= 0;
    end
    else begin
      kern_valid      <= bram_valid;
      kern_has_bias   <= bram_has_bias;
    end
    kern_activations  <= {{`FC0_NEURONS{bram_activations[1]}}, {`FC0_NEURONS{bram_activations[0]}}};
    kern_bias         <= 0;//bias;
    kern_neuron_id    <= {2{neuron_id}};
    weights           <= {data_out_b, data_out_a};
  end
  
   
  assign kern_mult1   =   (forward) ? weights      : b_kern_grad;
  
  assign kern_mult2   =   (forward) ? kern_activations : b_kern_act;
     
  // Computational kernel for the fully connected layer  
  genvar i;
  generate
    for (i = 0; i < `FC0_N_KERNELS; i=i+1) begin
      fc_kernel #(.FAN_IN(`FC0_KERNEL_FAN_IN), .ID_WIDTH(7)) fc_kernel_i (
        // input
        .clk(clk),
        .rst(rst),
        .activation_i(kern_mult2[i]),
        .weight(kern_mult1[i]),
        .bias(18'b0),
        .neuron_id_i(kern_neuron_id[i]),
        .has_bias(kern_has_bias),
        .valid_i(kern_valid),
        .b_valid_i(b_valid[2]),
        .bp_mode(bp_mode),
        // output
        .b_gradient_o(b_kern_grad_o[i]),
        .b_valid_o(b_kern_valid_o[i]),
        .activation_o(kern_activation_o[i]),
        .neuron_id_o(kern_neuron_id_o[i]),
        .valid_o(valid[i])
      );
    end
  endgenerate  
  
  bit [7: 0] b;
  always_ff @(posedge clk) begin
    if (&valid) begin
      for (b = 0; b < `FC0_NEURONS; b = b + 1) begin
        act_o_sign[neuron_id_o[b]]   <= activation_o_rel[b][`PREC - 1];
      end
    end
  end
  
 
  assign valid_act_o  = &valid;
  assign neuron_id_o  = kern_neuron_id_o[`FC0_NEURONS - 1: 0];
  
  bit [8: 0] m, n;
  always_comb begin
    for (m = 0, n = `FC0_NEURONS; m < `FC0_NEURONS; m=m+1, n=n+1) begin
      activation_o_rel[m] = $signed(kern_activation_o[m]) + $signed(kern_activation_o[n]);
      activation_o[m] = activation_o_rel[m][`PREC - 1] ? 0 : activation_o_rel[m];
    end
  end
  
     
   bit [7: 0] q, w;
  // Backward pass logic
  always_ff @(posedge clk) begin
    for (q = 0, w = `FC0_NEURONS; q < `FC0_NEURONS; q = q + 1, w = w+1) begin
      b_gradient[q]   <= act_o_sign[q] ? 0 : b_gradient_i[q];
      b_gradient[w]   <= act_o_sign[q] ? 0 : b_gradient_i[q];
    end
    b_gradient_pl <= b_gradient;
    b_kern_grad   <= b_gradient_pl;      
    
    b_act         <= b_activation_i;
    b_act_pl      <= b_act;
    b_kern_act    <= b_act_pl;      
    
    
    b_act_id      <= {b_act_id[2:0], b_activation_id};
    b_valid       <= {b_valid[1: 0], b_valid_i};
  end
endmodule