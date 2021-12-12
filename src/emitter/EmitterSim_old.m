classdef EmitterSim_old < handle
    properties
        % Arguments
        d {mustBeNumeric}           % tip-to-extractor distance [m]
        rc {mustBeNumeric}          % radius of curvature of tip [m]
        alpha {mustBeNumeric}       % cone half-angle [rad]
        h {mustBeNumeric}           % height of emitter [m]
        ra {mustBeNumeric}          % radius of emitter aperture [m]
        
        % Internal
        rb                          % base radius of cone [m]
        geo                         % geometry matrix
        filename                    % string filename
        emagmodel                   % electrostatic model
        external_face               % Geometry face IDs
        emitter_face                %
        air_face                    %
        extractor_face              %
        results                     % ElectrostaticResults
        xg                          % X interpolation grid
        yg                          % Y interpolation grid
        EintrpX                     % Ex on interpolation grid
        EintrpY                     % Ey on interpolation grid
        emitterSolution             % E field along emitter surface
        cmap                        % color map for plots

        far_field                   % for testing (switch geometry)
        N

    end

    methods(Static)
        function [exp, f] = get_sig_figs(arr, num_digits)
            exp = floor(log10(arr));
            f = fix(abs(arr) .* 10 .^ (-floor(log10(abs(arr))) + num_digits-1));
        end

        % Generate unique filename based on geometry parameters
        function filename = get_filename(d, rc, alpha, h, ra)
            spec = '%.2E';
            filename = strcat(['esim_d_', num2str(d,spec)],...
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
    end

    methods
        % Constructor
        function self = EmitterSim(d, rc, alpha, h, ra, far_field)
            self.d = d;
            self.rc = rc;
            self.alpha = alpha;
            self.h = h;
            self.ra = ra;
            self.filename = self.get_filename(self.d, self.rc, self.alpha, self.h, self.ra);
            self.far_field = far_field; % specify what (version) geometry to build

            % Color map
            self.cmap = [128 0	0
                        230	25	75
                        245	130	48
                        255	255	25
                        210	245	60
                        115	250	80
                        170	255	195
                        70	240	240
                        0	130	200
                        0	0	128];

            % Face IDs
            self.external_face = 0;
            self.emitter_face = 1;
            self.air_face = 2;
            self.extractor_face = 3;
            self.N = 100;

            % Initialization setup
            if self.far_field == 1
                self.farFieldGeometry();
            elseif self.far_field == 2
                self.axiGeometry() % This is the current best one
            elseif self.far_field == 3
                self.hyperboloid();
            else
                self.constructGeometry(); % This is the OG
            end
            self.setupModel();
        end

        function hyperboloid(self)
            % Calculated values
            curve_y_delta = self.rc - self.rc*sin(self.alpha);  % curved edge y-delta [m]
            curve_x_delta = self.rc*cos(self.alpha);            % curved edge x-delta [m]
            h_cone = self.h - curve_y_delta;                    % height of emitter cone [m]
            self.rb = curve_x_delta + h_cone*tan(self.alpha);   % Emitter base radius [m]
        
            % Emitter points [X,Y] : (0,0) is below emitter tip
            E1 = [0 self.d];
            E3 = [E1(1)+self.rb, E1(2)+self.h];
            x_coord = linspace(E1(1),E3(1),self.N+1);
            H0 = 1/self.rc;
            eta0 = sqrt( (H0*self.d)/(H0*self.d + 1) );
            a = self.d * sqrt( (H0*self.d + 1)/(H0*self.d) );
            xi_coord = sqrt(1 + (x_coord/(a*sqrt(1-eta0^2))).^2);
            y_coord = a*eta0*xi_coord;
            E = zeros(self.N,10);
            E(:,1) = 2;
            E(:,2) = x_coord(1:self.N);
            E(:,3) = x_coord(2:self.N+1);
            E(:,4) = y_coord(1:self.N);
            E(:,5) = y_coord(2:self.N+1);
            E(:,6) = self.external_face;
            E(:,7) = self.emitter_face;
            x_max = E3(1) * 1.2;          % Scene x boundary
            E3(2) = max(y_coord);
           
            % Bounding box (scene) points
            S1 = [0 0];
            S2 = [x_max S1(2)];
            S3 = [S2(1) E3(2)];

            % Construct decomposed geometry matrix manually (descg as an alternative)
            % Format: [segment_type, start_x, end_x, start_y, end_y, left_face_id, right_face_id, center_x, center_y, radius]'
            % segment_type: 1=round edge, 2=straight line
            air_top = [2, E3(1), S3(1), E3(2), S3(2), self.external_face, self.emitter_face, zeros(1,3)]';
            air_right = [2, S3(1), S2(1), S3(2), S2(2), self.external_face, self.emitter_face, zeros(1,3)]';
            bottom = [2, S2(1), S1(1), S2(2), S1(2), self.external_face, self.emitter_face, zeros(1,3)]';
            air_left = [2, S1(1), E1(1), S1(2), E1(2), self.external_face, self.emitter_face, zeros(1,3)]';
        
            self.geo = [E', air_top, air_right, bottom, air_left]; 
        end

        function axiGeometry(self)
            % Constants
            extractor_thickness = 0.001;%50e-6;
            far_field_location = min(0,self.d) - (self.h/2) - extractor_thickness;
%             extractor_thickness = 1e-3;  % [m]
%             far_field_location = min(0,self.d) - 10e-3 - extractor_thickness;
           
            % Calculated values
            curve_y_delta = self.rc - self.rc*sin(self.alpha);  % curved edge y-delta [m]
            curve_x_delta = self.rc*cos(self.alpha);            % curved edge x-delta [m]
            h_cone = self.h - curve_y_delta;                    % height of emitter cone [m]
            self.rb = curve_x_delta + h_cone*tan(self.alpha);   % Emitter base radius [m]
            x_max = self.rb * 1.2;                              % Scene x boundary
        
            % Emitter points [X,Y] : (0,0) is below emitter tip
            E1 = [0 self.d];
            Ecenter = [E1(1), E1(2)+self.rc];
            E2 = [E1(1)+curve_x_delta, E1(2)+curve_y_delta];
            E3 = [E1(1)+self.rb, E1(2)+self.h];
        
            % Bounding box (scene) points
            S1 = [0 far_field_location];
            S2 = [x_max S1(2)];
            S3 = [S2(1) E3(2)];
        
            % Extractor plate points (rectangle)
            P1 = [self.ra -extractor_thickness];
            P2 = [x_max P1(2)];
            P3 = [P2(1) 0];
            P4 = [P1(1) P3(2)];

            % Construct decomposed geometry matrix manually (descg as an alternative)
            % Format: [segment_type, start_x, end_x, start_y, end_y, left_face_id, right_face_id, center_x, center_y, radius]'
            % segment_type: 1=round edge, 2=straight line
            round_edge = [1, E1(1), E2(1), E1(2), E2(2), self.external_face, self.emitter_face, Ecenter, self.rc]';
            cone_slant = [2, E2(1), E3(1), E2(2), E3(2), self.external_face, self.emitter_face, zeros(1,3)]';

            air_top = [2, E3(1), S3(1), E3(2), S3(2), self.external_face, self.emitter_face, zeros(1,3)]';
            air_right = [2, S3(1), P3(1), S3(2), P3(2), self.external_face, self.emitter_face, zeros(1,3)]';

            extractor_top = [2, P3(1), P4(1), P3(2), P4(2), self.external_face, self.emitter_face, zeros(1,3)]';
            extractor_left = [2, P4(1), P1(1), P4(2), P1(2), self.external_face, self.emitter_face, zeros(1,3)]';
            extractor_bottom = [2, P1(1), P2(1), P1(2), P2(2), self.external_face, self.emitter_face, zeros(1,3)]';

            air_low_right = [2, P2(1), S2(1), P2(2), S2(2), self.external_face, self.emitter_face, zeros(1,3)]';
            air_bottom = [2, S2(1), S1(1), S2(2), S1(2), self.external_face, self.emitter_face, zeros(1,3)]';
            air_left = [2, S1(1), E1(1), S1(2), E1(2), self.external_face, self.emitter_face, zeros(1,3)]';
        
            self.geo = [round_edge, cone_slant, air_top, air_right,...
                extractor_top, extractor_left, extractor_bottom,...
                air_low_right, air_bottom, air_left]; 
        end

        function constructGeometry(self)
            % Constants
            extractor_thickness = 1e-3;  % [m]
            
            % Calculated values
            curve_y_delta = self.rc - self.rc*sin(self.alpha);  % curved edge y-delta [m]
            curve_x_delta = self.rc*cos(self.alpha);            % curved edge x-delta [m]
            h_cone = self.h - curve_y_delta;                    % height of emitter cone [m]
            self.rb = curve_x_delta + h_cone*tan(self.alpha);   % Emitter base radius [m]
            x_max = self.rb * 1.1;                              % Scene x boundary
        
            % Emitter points [X,Y] : (0,0) below emitter tip
            E1 = [0 self.d];
            Ecenter = [E1(1), E1(2)+self.rc];
            E2 = [E1(1)+curve_x_delta, E1(2)+curve_y_delta];
            E3 = [E1(1)+self.rb, E1(2)+self.h];
            E4 = [E1(1), E1(2)+self.h];
        
            % Bounding box (scene) points
            S1 = [0 -extractor_thickness];
            S2 = [x_max S1(2)];
            S3 = [S2(1) E3(2)];
            S4 = E4;
        
            % Extractor plate points (rectangle)
            P1 = [self.ra -extractor_thickness];
            P2 = [x_max P1(2)];
            P3 = [P2(1) 0];
            P4 = [P1(1) P3(2)];

            % Construct decomposed geometry matrix manually (descg as an alternative)
            % Format: [segment_type, start_x, end_x, start_y, end_y, left_face_id, right_face_id, center_x, center_y, radius]'
            % segment_type: 1=round edge, 2=straight line
        
            % Emitter
            round_edge = [1, E1(1), E2(1), E1(2), E2(2), self.emitter_face, self.air_face, Ecenter, self.rc]';
            cone_slant = [2, E2(1), E3(1), E2(2), E3(2), self.emitter_face, self.air_face, zeros(1,3)]';
            emitter_base = [2, E3(1), E4(1), E3(2), E4(2), self.emitter_face, self.external_face, zeros(1,3)]';
            emitter_center = [2, E4(1), E1(1), E4(2), E1(2), self.emitter_face, self.external_face, zeros(1,3)]';
            emitter = [round_edge, cone_slant, emitter_base, emitter_center]; % Edges 1-4
        
            % Extractor
            bottom = [2, P1(1), P2(1), P1(2), P2(2), self.extractor_face, self.external_face, zeros(1,3)]';
            right = [2, P2(1), P3(1), P2(2), P3(2), self.extractor_face, self.external_face, zeros(1,3)]';
            top = [2, P3(1), P4(1), P3(2), P4(2), self.extractor_face, self.air_face, zeros(1,3)]';
            left = [2, P4(1), P1(1), P4(2), P1(2), self.extractor_face, self.air_face, zeros(1,3)]';
            extractor = [bottom, right, top, left]; % Edges 5-8
        
            % Air
            air_left = [2, E1(1), S1(1), E1(2), S1(2), self.air_face, self.external_face, zeros(1,3)]';
            air_bottom = [2, S1(1), P1(1), S1(2), P1(2), self.air_face, self.external_face, zeros(1,3)]';
            air_top = [2, E3(1), S3(1), E3(2), S3(2), self.external_face, self.air_face, zeros(1,3)]';
            air_right = [2, S3(1), P3(1), S3(2), P3(2), self.external_face, self.air_face, zeros(1,3)]';
            air = [air_left, air_bottom, air_top, air_right]; % Edges 9-12
        
            self.geo = [emitter, extractor, air]; % decomposed geometry matrix
        end

        function farFieldGeometry(self)
            % Constants
            extractor_thickness = 1e-3;  % [m]
            far_field_distance = 10e-3;  % [m]
            
            % Calculated values
            curve_y_delta = self.rc - self.rc*sin(self.alpha);  % curved edge y-delta [m]
            curve_x_delta = self.rc*cos(self.alpha);            % curved edge x-delta [m]
            h_cone = self.h - curve_y_delta;                    % height of emitter cone [m]
            self.rb = curve_x_delta + h_cone*tan(self.alpha);   % Emitter base radius [m]
            x_max = self.rb * 1.1;                              % Scene x boundary
        
            % Emitter points [X,Y] : (0,0) below emitter tip
            E1 = [0 self.d];
            Ecenter = [E1(1), E1(2)+self.rc];
            E2 = [E1(1)+curve_x_delta, E1(2)+curve_y_delta];
            E3 = [E1(1)+self.rb, E1(2)+self.h];
            E4 = [E1(1), E1(2)+self.h];
        
            % Bounding box (scene) points
            S1 = [0 -extractor_thickness-far_field_distance];
            S2 = [x_max S1(2)];
            S3 = [S2(1) E3(2)];
            S4 = E4;
        
            % Extractor plate points (rectangle)
            P1 = [self.ra -extractor_thickness];
            P2 = [x_max P1(2)];
            P3 = [P2(1) 0];
            P4 = [P1(1) P3(2)];

            % Construct decomposed geometry matrix manually (descg as an alternative)
            % Format: [segment_type, start_x, end_x, start_y, end_y, left_face_id, right_face_id, center_x, center_y, radius]'
            % segment_type: 1=round edge, 2=straight line
        
            % Emitter
            round_edge = [1, E1(1), E2(1), E1(2), E2(2), self.emitter_face, self.air_face, Ecenter, self.rc]';
            cone_slant = [2, E2(1), E3(1), E2(2), E3(2), self.emitter_face, self.air_face, zeros(1,3)]';
            emitter_base = [2, E3(1), E4(1), E3(2), E4(2), self.emitter_face, self.external_face, zeros(1,3)]';
            emitter_center = [2, E4(1), E1(1), E4(2), E1(2), self.emitter_face, self.external_face, zeros(1,3)]';
            emitter = [round_edge, cone_slant, emitter_base, emitter_center]; % Edges 1-4
        
            % Extractor
            bottom = [2, P1(1), P2(1), P1(2), P2(2), self.extractor_face, self.air_face, zeros(1,3)]';
            right = [2, P2(1), P3(1), P2(2), P3(2), self.extractor_face, self.external_face, zeros(1,3)]';
            top = [2, P3(1), P4(1), P3(2), P4(2), self.extractor_face, self.air_face, zeros(1,3)]';
            left = [2, P4(1), P1(1), P4(2), P1(2), self.extractor_face, self.air_face, zeros(1,3)]';
            extractor = [bottom, right, top, left]; % Edges 5-8
        
            % Air
            air_left = [2, E1(1), S1(1), E1(2), S1(2), self.air_face, self.external_face, zeros(1,3)]';
            air_bottom = [2, S1(1), S2(1), S1(2), S2(2), self.air_face, self.external_face, zeros(1,3)]';
            air_top = [2, E3(1), S3(1), E3(2), S3(2), self.external_face, self.air_face, zeros(1,3)]';
            air_right = [2, S3(1), P3(1), S3(2), P3(2), self.external_face, self.air_face, zeros(1,3)]';
            air_low_right = [2, P2(1), S2(1), P2(2), S2(2), self.external_face, self.air_face, zeros(1,3)]';
            air = [air_left, air_bottom, air_top, air_right, air_low_right]; % Edges 9-13
        
            self.geo = [emitter, extractor, air]; % decomposed geometry matrix
        end

        function [xmin, xmax, ymin, ymax] = get_geo_limits(self)
            [max_vals, ~] = max(self.emagmodel.Geometry.Vertices);
            xmax = max_vals(1); ymax = max_vals(2);
            [min_vals, ~] = min(self.emagmodel.Geometry.Vertices);
            xmin = min_vals(1); ymin = min_vals(2);
        end

        function setupModel(self)
            % Setup Electrostatic simulation
            self.emagmodel = createpde('electromagnetic', 'electrostatic-axisymmetric'); 
            
            % Constants
            emitter_voltage = 1000;    % [V]
            permittivity_air = 1;        % Relative permittivity [-]
            permittivity_pec = 1000000;    % Relative permittivity [-]
            
            % Add geometry
            ag = geometryFromEdges(self.emagmodel, self.geo); % analytic geometry properties
            
            % Set Electrostatic properties and BCs
            self.emagmodel.VacuumPermittivity = 8.8541878128E-12;
            
            if self.far_field == 1
                electromagneticBC(self.emagmodel, 'Voltage', emitter_voltage, 'Edge', [faceEdges(ag,self.emitter_face), 11]);
                electromagneticBC(self.emagmodel, 'Voltage', 0, 'Edge', [faceEdges(ag,self.extractor_face), 10, 13]);
                electromagneticSource(self.emagmodel, 'ChargeDensity', 0, 'Face', self.air_face);
                electromagneticProperties(self.emagmodel, 'RelativePermittivity', permittivity_air, 'Face', self.air_face);
                electromagneticProperties(self.emagmodel, 'RelativePermittivity', permittivity_pec, 'Face', [self.emitter_face, self.extractor_face]);
            elseif self.far_field == 2
                electromagneticBC(self.emagmodel, 'Voltage', emitter_voltage, 'Edge', [1,2,3]);
                electromagneticBC(self.emagmodel, 'Voltage', 0, 'Edge', [5,6,7,9]);
                electromagneticSource(self.emagmodel, 'ChargeDensity', 0, 'Face', self.emitter_face);
                electromagneticProperties(self.emagmodel, 'RelativePermittivity', permittivity_air, 'Face', self.emitter_face);
            elseif self.far_field == 3
                electromagneticBC(self.emagmodel, 'Voltage', emitter_voltage, 'Edge', [1:self.N+1]);
                electromagneticBC(self.emagmodel, 'Voltage', 0, 'Edge', [self.N+3]);
                electromagneticSource(self.emagmodel, 'ChargeDensity', 0, 'Face', self.emitter_face);
                electromagneticProperties(self.emagmodel, 'RelativePermittivity', permittivity_air, 'Face', self.emitter_face);
            else
                electromagneticBC(self.emagmodel, 'Voltage', emitter_voltage, 'Edge', [faceEdges(ag,self.emitter_face), 11]);
                electromagneticBC(self.emagmodel, 'Voltage', 0, 'Edge', faceEdges(ag,self.extractor_face));
                electromagneticSource(self.emagmodel, 'ChargeDensity', 0, 'Face', self.air_face);
                electromagneticProperties(self.emagmodel, 'RelativePermittivity', permittivity_air, 'Face', self.air_face);
                electromagneticProperties(self.emagmodel, 'RelativePermittivity', permittivity_pec, 'Face', [self.emitter_face, self.extractor_face]);
            end
            
        end

        function emitterMesh(self, mesh_size, refine_factor, growth_rate)
            arguments
                self
                mesh_size (1,1) {mustBeNumeric} = 0
                refine_factor (1,1) {mustBeNumeric} = 1
                growth_rate (1,1) {mustBeNumeric} = 1.1
            end
            if mesh_size == 0
                generateMesh(self.emagmodel);
            else
                if self.far_field == 2
                    generateMesh(self.emagmodel, 'Hmax', mesh_size, 'Hgrad', growth_rate, 'Hedge', {[1 2 5 6 7 10], mesh_size/refine_factor});
                elseif self.far_field == 3
                    generateMesh(self.emagmodel, 'Hmax', mesh_size, 'Hgrad', growth_rate, 'Hedge', {[1:self.N+1], mesh_size/refine_factor});
                else
                    % Refine around edges 1, 2, 7, 8, 9, 10
                    generateMesh(self.emagmodel, 'Hmax', mesh_size, 'Hgrad', growth_rate, 'Hedge', {[1 2 7 8 9 10], mesh_size/refine_factor});
                end
                
            end
        end

        function emitterSolve(self)
            self.results = solve(self.emagmodel);
            [xmin, xmax, ymin, ymax] = self.get_geo_limits();

            % Interpolate results
            mesh_size = self.emagmodel.Mesh.MinElementSize;
            Nx = ceil((xmax-xmin)/mesh_size) + 1;
            Ny = ceil((ymax-ymin)/mesh_size) + 1;
            xmesh = linspace(xmin, xmax, Nx);
            ymesh = linspace(ymin, ymax, Ny);
            [self.xg, self.yg] = meshgrid(xmesh, ymesh);
            Eintrp = interpolateElectricField(self.results, self.xg, self.yg);
            self.EintrpX = reshape(Eintrp.Ex,size(self.xg));
            self.EintrpY = reshape(Eintrp.Ey,size(self.yg));
        
            % Get emitter nodes
            if self.far_field == 3
                emitter_nodes = findNodes(self.emagmodel.Mesh, 'region', 'Edge', [1:self.N]);
            else
                emitter_nodes = findNodes(self.emagmodel.Mesh, 'region', 'Edge', [1 2]);
            end
            emitter_x = self.emagmodel.Mesh.Nodes(1,emitter_nodes);
            emitter_y = self.emagmodel.Mesh.Nodes(2,emitter_nodes);
            emitter_ds = [0, sqrt(diff(emitter_x).^2 + diff(emitter_y).^2)];
            emitter_s = cumsum(emitter_ds);
            emitter_Ex = self.results.ElectricField.Ex(emitter_nodes)';
            emitter_Ey = self.results.ElectricField.Ey(emitter_nodes)';
            self.emitterSolution = [emitter_x' emitter_y' emitter_s' emitter_Ex' emitter_Ey'];
        end

        function emitterPlot(self)
            % Plot settings
            set(groot,'defaultAxesTickLabelInterpreter','latex');
            set(groot,'defaulttextinterpreter','latex');
            set(groot,'defaultlegendInterpreter','latex');
            n = 100;
            rgb_cmap = self.interp_cmap(self.cmap, n);

            % Electric field at emitter
            figure()
            emitter_s = self.emitterSolution(:,3);
            emitter_Emag = sqrt(self.emitterSolution(:,4).^2 + self.emitterSolution(:,5).^2);
            [E_max, idx] = max(emitter_Emag);
            s_max = emitter_s(idx);
            set(gcf,'color','white');
            plot(emitter_s*1000, emitter_Emag,'ok');
            hold on
            plot(s_max, E_max, '*g');
            legend('Emitter nodes', 'Max $\left| \vec{E} \right|$');
            xlabel('Distance along emitter from tip to base [mm]');
            ylabel('Electric field magnitude $\left[ \frac{V}{m} \right]$');
        
            fig = figure();
            fig.Position = [0,0,900, 500]; % 1000, 500
            set(gcf,'color','white');
        
            % Mesh
            h1 = subplot(1,2,1);
            pdemesh(self.emagmodel);
            originalSize1 = get(gca, 'Position');
            axis tight;
            xlabel('$x$ [m]');
            ylabel('$y$ [m]');
%             title_str = sprintf('Mesh size = %.2f mm',self.emagmodel.Mesh.MaxElementSize*1000);
%             th = title(title_str);
%             th.Position(2) = th.Position(2) * 1.05;
        
            % Interpolated grid
        %     h2 = subplot(1,3,2);
        %     h = pdegplot(emagmodel);
        %     originalSize2 = get(gca, 'Position');
        %     h.LineWidth = 1.5;
        %     h.Color = [1 0 0];
        %     xlabel('$x$ [m]','Interpreter','latex');
        %     ylabel('$y$ [m]','Interpreter','latex');
        %     title_str = sprintf('Interpolated grid size = %.2f mm',delta*1000);
        %     th = title(title_str);
        %     th.Position(2) = th.Position(2) * 1.05;
        %     hold on;
        %     plot(xg,yg,'.k');
        
            % Electric field magnitude
            h3 = subplot(1,2,2);
            Emag = sqrt(self.results.ElectricField.Ex.^2 + ...
                self.results.ElectricField.Ey.^2);
            pdeplot(self.emagmodel,'XYData',Emag');
%             [~,c] = contourf(self.emagmodel.Mesh.Nodes(1,:), ...
%                 self.emagmodel.Mesh.Nodes(2,:), Emag, n);
%             Emag = sqrt(self.EintrpX.^2 + self.EintrpY.^2);
%             [~,c] = contourf(self.xg, self.yg, Emag, n);
            originalSize3 = get(gca, 'Position');
            c.LineColor = 'none';
            colormap(flipud(rgb_cmap));
            cb = colorbar();
            cb.TickLabelInterpreter = 'latex';
            cb.LineWidth = 0.8;
            cb.FontSize = 11;
            set(get(cb,'label'),'interpreter','latex');
            set(get(cb,'label'),'string','Electric field magnitude [$\frac{V}{m}$]');
            set(get(cb,'label'),'FontSize',11);
            xlabel('$x$ [m]','Interpreter','latex');
            ylabel('$y$ [m]','Interpreter','latex');
            hold on;
            plot_h = pdegplot(self.emagmodel);
            plot_h.LineWidth = 1.5;
            plot_h.Color = [1 0 0];
        
            set(h1, 'Position', originalSize1);
        %     set(h2, 'Position', originalSize2);
            set(h3, 'Position', originalSize3);
        end

    end
end