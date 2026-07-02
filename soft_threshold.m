function C = soft_threshold(qian, hou)
    % SSC求法：
    % α||C||_1     s.t.  X=C  求C    
    % 可写为以下式子：
    % α||C||_1 + μ/2||  X-C+J/μ     -     α/μ   ||_F^2
    % 或
    % α||C||_1 + <J, X-C> + μ/2||X-C||_F^2
    % 其中，
    % qian = X-C+J/μ
    % hou = α/μ
    C = sign(qian)  .* max(abs(qian)-hou, 0 );
end