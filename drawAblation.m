y=[73.40 73.99 73.21 74.05; 32.80 34.07 32.57 34.21; 71.67 75.32 70.99 75.38; 34.73 34.53 33.78 37.38; 47.11 52.19 43.64 58.68; 67.68 71.29 65.14 72.13];
% 四个变种（数据集1）；四个变种（数据集2）；
b=bar(y);
grid on;
ch = get(b,'children');
set(gca,'XTickLabel',{'AR','FERET','ISOLET','Letter-rec','MNIST','USPS'})
% set(gca,'yticklabel',{'0.8','0.85','0.9','0.95','1','1.05','1.1','1.15'});
% set(ch,'FaceVertexCData',[1 0 1;0 0 0;])
legend('AWDR_W','AWDR_P','AWDR_{W,P}','AWDR(Ours)');
xlabel('Datasets');
ylabel('Accuracy(%)');