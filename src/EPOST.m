classdef EPOST
    % Static helper methods for post-processing emitter solutions
    properties (Constant)
        cmap = [128 0	0
                230	25	75
                245	130	48
                255	255	25
                210	245	60
                115	250	80
                170	255	195
                70	240	240
                0	130	200
                0	0	128];
    end

    methods (Static)
        function [exp, f] = get_sig_figs(arr, num_digits)
            exp = floor(log10(arr));
            f = fix(abs(arr) .* 10 .^ (-floor(log10(abs(arr))) + num_digits-1));
        end

        % Generate unique filename based on geometry parameters
        function filename = get_filename(d, rc, alpha, h, ra, prefix)
            spec = '%.2E';
            filename = strcat([prefix, '_d_', num2str(d,spec)],...
               ['_rc_', num2str(rc,spec)],...
               ['_a_', num2str(alpha,spec)],...
               ['_h_', num2str(h,spec)],...
               ['_ra_', num2str(ra,spec), '.mat']);
        end

        % Interpolate colormap to n points
        function int_cmap = interp_cmap(cmap, n)
            num_colors = size(cmap,1);
            x = 1:1:num_colors;
            xq = linspace(1,num_colors,n);
            red = cmap(:,1);
            green = cmap(:,2);
            blue = cmap(:,3);
            red_vq = interp1(x,red,xq);
            green_vq = interp1(x,green,xq);
            blue_vq = interp1(x,blue,xq);
            int_cmap = [red_vq',green_vq',blue_vq']./255;
        end

        function [xmin, xmax, ymin, ymax] = get_geo_limits(emitter)
            [max_vals, ~] = max(emitter.emagmodel.Geometry.Vertices);
            xmax = max_vals(1); ymax = max_vals(2);
            [min_vals, ~] = min(emitter.emagmodel.Geometry.Vertices);
            xmin = min_vals(1); ymin = min_vals(2);
        end

        function [x,y,s,Ex,Ey] = emitter_solution(emitter)
            emitter_nodes = findNodes(emitter.emagmodel.Mesh, 'region', 'Edge', emitter.e_edges);
            x = emitter.emagmodel.Mesh.Nodes(1,emitter_nodes);
            y = emitter.emagmodel.Mesh.Nodes(2,emitter_nodes);
            ds = [0, sqrt(diff(x).^2 + diff(y).^2)];
            s = cumsum(ds);
            Ex = emitter.emagresults.ElectricField.Ex(emitter_nodes)';
            Ey = emitter.emagresults.ElectricField.Ey(emitter_nodes)';
        end

        function [ypoints, Ex, Ey] = ms_solution(rc, d, V0, xpoints)
            H0 = 1/rc;
            eta0 = sqrt( (H0*d)/(H0*d + 1) );
            a = d * sqrt( (H0*d + 1)/(H0*d) );
            xi_coord = sqrt(1 + (xpoints/(a*sqrt(1-eta0^2))).^2);
            e_premult = V0/(a*atanh(eta0));
            r_premult = (eta0./(xi_coord.^2 - eta0^2)) .* sqrt((xi_coord.^2 - 1)/(1-eta0^2));
            z_premult = xi_coord ./ (xi_coord.^2 - eta0^2);
            Ex = e_premult*r_premult;
            Ey = e_premult*(-z_premult);
            ypoints = a*xi_coord*eta0;
        end

        function h = pdeplot(emitter)
%             set(groot,'defaultAxesTickLabelInterpreter','latex');
%             set(groot,'defaulttextinterpreter','latex');
%             set(groot,'defaultlegendInterpreter','latex');
            rgb_cmap = EPOST.interp_cmap(EPOST.cmap, 100);
            Emag = sqrt(emitter.emagresults.ElectricField.Ex.^2 + ...
                emitter.emagresults.ElectricField.Ey.^2);
            pdeplot(emitter.emagmodel,'XYData',Emag');
            originalSize = get(gca, 'Position');
            set(gcf,'color','white');
            colormap(flipud(rgb_cmap));
            cb = colorbar();
            cb.LineWidth = 0.8;
            cb.FontSize = 11;
            set(get(cb,'label'),'interpreter','latex');
            set(get(cb,'label'),'string','Electric field magnitude [$\frac{V}{m}$]');
            set(get(cb,'label'),'FontSize',11);
            xlabel('$x$ [m]','Interpreter','latex');
            ylabel('$y$ [m]','Interpreter','latex');
            hold on;
            plot_h = pdegplot(emitter.emagmodel);
            plot_h.LineWidth = 1.5;
            plot_h.Color = [1 0 0];
            set(gca,'Position',originalSize);
            h = gca;
            hold off;
        end

        function h = pdemesh(emitter)
            pdemesh(emitter.emagmodel);
            axis tight;
            xlabel('$x$ [m]','Interpreter','latex');
            ylabel('$y$ [m]','Interpreter','latex');
            set(gcf,'color','white');
            h = gca;
        end

        function solplot(emitter)
            figure()
            subplot(1,2,1);
            EPOST.pdemesh(emitter);
            box off
            subplot(1,2,2);
            EPOST.pdeplot(emitter);
        end

        function emitterplot(emitter)
            [~,~,s,Ex,Ey] = EPOST.emitter_solution(emitter);
            pct_s = s./s(end)*100;
            figure()
            Emag = sqrt(Ex.^2 + Ey.^2);
            set(gcf,'color','white');
            plot(pct_s, Emag,'-b');
            xlabel('Percent distance along emitter (tip to base)',...
                'Interpreter','latex');
            ylabel('Electric field magnitude $\left[ \frac{V}{m} \right]$',...
                'Interpreter','latex');
        end

        function vectorplot(emitter, s)
            x = emitter.emagmodel.Mesh.Nodes(1,:);
            y = emitter.emagmodel.Mesh.Nodes(2,:);
            ex = emitter.emagresults.ElectricField.Ex';
            ey = emitter.emagresults.ElectricField.Ey';
            mag = sqrt(ex.^2 + ey.^2);
            ux = ex./mag;
            uy = ey./mag;

            figure()
            plot_h = pdegplot(emitter.emagmodel);
            plot_h.LineWidth = 1.5;
            plot_h.Color = [1 0 0];
            hold on;
            quiver(x(1:s:end), y(1:s:end), ux(1:s:end), uy(1:s:end),0.6,'k');
            hold off;
            [xmin, xmax, ymin, ymax] = EPOST.get_geo_limits(emitter);
            xlim([xmin, xmax]);
            ylim([ymin, ymax]);
        end

    end
end