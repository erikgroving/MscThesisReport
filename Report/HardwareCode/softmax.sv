`timescale 1ns / 1ps

module softmax(
  input                                               clk,
  input                                               reset,
  input                                               start,
  input [`PREC - 1: 0]                                max,
  input [`FC2_NEURONS - 1: 0][`PREC - 1: 0]           act_in,
  
  output logic                                        valid_o,
  output logic [`FC2_NEURONS - 1: 0][`PREC - 1: 0]    grad_o
  );
  
  
  logic [`FC2_NEURONS - 1: 0][23: 0]  act_in_norm;
  logic [`FC2_NEURONS - 1: 0][23: 0]  fixed_exp_res;
  logic [`FC2_NEURONS - 1: 0][31: 0]  act_in_norm_float;
  logic [31: 0]                       float_o;
  logic [31: 0]                       float_exp_o;
  logic                               float_valid_o;
  logic                               float_exp_valid_o;
  logic [23: 0]                       fixed_exp_o;
  logic [23: 0]                       fixed_exp_sum;
  logic                               fixed_exp_valid_o;
  logic [3: 0]                        fp_in_ptr;
  logic [3: 0]                        fixed_exp_ptr;
  logic [3: 0]                        div_ptr;
  logic                               in_prog;
  logic [47: 0]                       div_o;
  logic                               div_valid_i;
  logic                               div_valid_o;
   
  bit [3: 0] i;
  always_ff @(posedge clk) begin
    if (start) begin
      for (i = 0; i < `FC2_NEURONS; i=i+1) begin
        act_in_norm[i]  <= $signed(act_in[i]) - $signed(max);
      end
    end
  end
  
  always_ff @(posedge clk) begin
    if (~in_prog) begin
      fp_in_ptr <= 4'b0;
    end
    if (in_prog && fp_in_ptr != `FC2_NEURONS) begin
      fp_in_ptr <= fp_in_ptr + 1'b1;
    end
  end
  
  fp_to_float fp_to_float_i (
    .s_axis_a_tdata(act_in_norm[fp_in_ptr]),
    .s_axis_a_tvalid(in_prog & fp_in_ptr != `FC2_NEURONS),
    .aclk(clk),
    
    .m_axis_result_tdata(float_o),
    .m_axis_result_tvalid(float_valid_o)
    
  );
  
  float_exp float_exp_i (
    .s_axis_a_tdata(float_o),
    .s_axis_a_tvalid(float_valid_o),
    .aclk(clk),
    
    .m_axis_result_tdata(float_exp_o),
    .m_axis_result_tvalid(float_exp_valid_o)
  );
  
  float_to_fp float_to_fp_i (
    .s_axis_a_tdata(float_exp_o),
    .s_axis_a_tvalid(float_exp_valid_o),
    .aclk(clk),
    
    .m_axis_result_tdata(fixed_exp_o),
    .m_axis_result_tvalid(fixed_exp_valid_o)  
  );
  
  always_ff @(posedge clk) begin
    if (~in_prog) begin
      fixed_exp_sum   <= 0;
    end
    if (fixed_exp_valid_o) begin
      fixed_exp_sum   <= $signed(fixed_exp_sum) + $signed(fixed_exp_o);
    end
  end
  
  always_ff @(posedge clk) begin
    if (~in_prog) begin
      fixed_exp_ptr           <= 4'b0;
      fixed_exp_res[fixed_exp_ptr]  <= 0;    
    end
    else if (fixed_exp_valid_o) begin
      fixed_exp_ptr           <= fixed_exp_ptr + 1'b1;
      fixed_exp_res[fixed_exp_ptr]  <= fixed_exp_o;
    end
  end
  
  
  always_ff @(posedge clk) begin
    if (~in_prog) begin
      div_ptr <= 0;
    end
    else if (div_ptr != `FC2_NEURONS && fixed_exp_ptr == `FC2_NEURONS) begin
      div_ptr <= div_ptr + 1'b1;
    end
  end
  
  assign div_valid_i = (fixed_exp_ptr == `FC2_NEURONS) & (div_ptr != `FC2_NEURONS);

  fixed_divider fixed_divider_i (
    .s_axis_divisor_tdata(fixed_exp_sum),
    .s_axis_divisor_tvalid(div_valid_i),
    
    .s_axis_dividend_tdata(fixed_exp_res[div_ptr]),
    .s_axis_dividend_tvalid(div_valid_i),
    
    .aclk(clk),

    .m_axis_dout_tdata(div_o),
    .m_axis_dout_tvalid(div_valid_o)
    
  );
  
  logic [3: 0] grad_ptr;
  always_ff @(posedge clk) begin
    if (~in_prog) begin
      grad_ptr  <= 0;
    end
    else if (div_valid_o) begin
      grad_ptr  <= grad_ptr + 1'b1;
    end
  end
  
  always_ff @(posedge clk) begin
    if (div_valid_o) begin
      grad_o[grad_ptr]  <= div_o[`PREC - 1: 0];
    end
    
    valid_o <= ((grad_ptr == `FC2_NEURONS) & in_prog);

  end
     
  always_ff @(posedge clk) begin
    if (reset) begin
      in_prog <= 1'b00;
    end
    else if (start) begin
      in_prog <= 1'b1;
    end
    else if (valid_o) begin
      in_prog <= 1'b0;
    end
  end
endmodule