clear all;
close all;
clc;
addpath('../emitter')
addpath('../postproc')

ra = linspace(1e-5, 4e-3, 25);
D0 = 360e-6;
Rc = 1.6e-5;
cha = 30*(pi/180);
h = 350e-6;
V = 100e3;
te = 76e-6;

emax = zeros(size(ra));
for i=1:length(ra)
    e = Emitter(D0, Rc, cha, h, ra(i), te, V);
    [x,y,s,Ex,Ey] = EPOST.emitter_solution(e);
    Emag = sqrt(Ex.^2 + Ey.^2);
    emax(i) = max(Emag);
end
[y, Ex, Ey] = EPOST.ms_solution(Rc, D0, V, 0);
ref_Emag = sqrt(Ex.^2 + Ey.^2);

figure()
semilogx(ra*1000, emax, '-or');
hold on;
yline(ref_Emag,'-k');
xlabel('Aperture radius $R_a$ [mm]','Interpreter','latex');
ylabel('E-field [V/m]', 'Interpreter','latex');
leg = legend('Simulation', 'Martinez-Sanchez');
set(leg, 'Interpreter','latex');