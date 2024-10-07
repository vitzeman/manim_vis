from manim import *

# INFO: Colors defined in the CTU corporate identity manual
CTU_BLUE = "#0065BD"
CTU_LIGHT_BLUE = "#6AADE4"
CTU_BLACK = "#000000"
CTU_GRAY = "#9B9B9B"
CTU_WHITE = "#FFFFFF"
CTU_OLIVE = "#A2AD00"
CTU_RED = "#C60C30"
CTU_ORANGE = "#E05206"
CTU_CYAN = "#00B2A9"
CTU_TURQUOISE = "#156570"
CTU_WINE = "#981F40"
CTU_PINK = "#F04D98"
CTU_YELLOW = "#F0AB00"

RICAIP_BLUE = "#213765"
RICAIP_GOLD = "#db9233"


darkmode = True
if darkmode:
    # bg_color = "#1f1f1f"
    bg_color = "#000000"
    fg_color = "#ffffff"
    # ray_color = YELLOW
    ray_color = CTU_BLUE
else:
    bg_color = "#ffffff"
    fg_color = "#000000"
    ray_color = CTU_BLUE


class PinHole(Scene):

    def construct(self):
        self.camera.background_color = bg_color

        # ORIGIN AND AXES
        c1 = Circle(radius=0.1, color=fg_color, fill_opacity=1)
        x_axis = Arrow(start=[-2, -2, 0], end=[2, 2, 0], color=fg_color, buff=0)
        y_axis = Arrow(start=[0, 0, 0], end=[0, 3.5, 0], color=fg_color, buff=0)
        z_axis = Arrow(start=[0, 0, 0], end=[9, 0, 0], color=fg_color, buff=0)

        c1_text = MathTex(r"\mathbf{C}", color=fg_color).next_to(c1, LEFT)
        ya_text = Text("Y", color=fg_color).next_to(y_axis.end, RIGHT)
        xa_text = Text("X", color=fg_color).next_to(x_axis.end, UP)
        za_text = Text("Z", color=fg_color).next_to(z_axis.end, RIGHT)

        coordinate_system = VGroup(
            c1, x_axis, y_axis, z_axis, c1_text, xa_text, ya_text, za_text
        )
        coordinate_system.shift([-4, 0, 0])
        # self.add(coordinate_system)

        # IMAGE PLANE
        img_plane = Rectangle(
            height=3, width=4, color=fg_color, fill_opacity=1.0, fill_color=bg_color
        )
        matrix = [[1, 0], [1, -1]]
        img_plane.apply_matrix(matrix)
        img_plane.move_to([0, 0, 0])
        print(DL, type(DL), DL[0], type(DL[0]))
        plane_text = MathTex(r"\Pi", color=fg_color).next_to(img_plane, DL, buff=0.1)

        c2 = Circle(radius=0.1, color=fg_color, fill_opacity=1)
        c2.move_to(img_plane.get_center())
        pp_text = MathTex(r"\mathbf{p}", color=fg_color).next_to(c2, DR, buff=0.1)
        za_add = Line(
            start=img_plane.get_center(),
            end=z_axis.end + [-4.5, 0, 0],
            color=fg_color,
            stroke_width=x_axis.stroke_width,
        )

        end = np.array([1.0, 1, 0])
        end /= np.linalg.norm(end)
        xi_axis = Arrow(
            start=[0, 0, 0],
            end=end,
            color=fg_color,
            buff=0,
            stroke_width=1,
            max_stroke_width_to_length_ratio=10,
            max_tip_length_to_length_ratio=0.1,
        )
        yi_axis = Arrow(
            start=[0, 0, 0],
            end=[0, 1.0, 0],
            color=fg_color,
            buff=0,
            stroke_width=1,
            max_stroke_width_to_length_ratio=10,
            max_tip_length_to_length_ratio=0.1,
        )
        xi_text = MathTex("x", color=fg_color).next_to(xi_axis.end, RIGHT, buff=0.1)
        yi_text = MathTex("y", color=fg_color).next_to(yi_axis.end, RIGHT, buff=0.1)
        dashed = DashedLine(
            start=img_plane.get_center(),
            end=[-1.5, 0, 0],
            color=fg_color,
            stroke_width=z_axis.stroke_width,
        )

        image_system = VGroup(
            img_plane,
            c2,
            za_add,
            pp_text,
            xi_axis,
            yi_axis,
            xi_text,
            yi_text,
            dashed,
            plane_text,
        )
        self.add(coordinate_system, image_system)

        # Point projection on image plane and camera center

        point = Dot([4, 2.7, 0], color=ray_color)
        point_text = MathTex(r"\mathbf{X}", color=ray_color).next_to(point, RIGHT)
        s = c1.get_center()
        g = point.get_center()
        point_line = DashedLine(s, g, color=ray_color)
        d = g - s
        d /= np.linalg.norm(d)

        inter = s + 5 * d

        print(inter)
        intersection = Cross(Dot(inter), stroke_color=ray_color, scale_factor=1.5)
        i_text = MathTex(r"\mathbf{x}", color=ray_color).next_to(
            intersection, UR, buff=0.1
        )
        # intersection = Cross(dot, color=YELLOW)

        point_system = VGroup(point, point_text, point_line, intersection, i_text)
        self.add(point_system)


class PinHoleVideo(Scene):
    def construct(self):

        self.camera.background_color = bg_color

        self.camera.center = [4.5, 0, 0]
        # ORIGIN AND AXES
        c1 = Circle(radius=0.1, color=fg_color, fill_opacity=1).shift([-4, 0, 0])
        x_axis = Arrow(start=[-2, -2, 0], end=[2, 2, 0], color=fg_color, buff=0).shift(
            [-4, 0, 0]
        )
        y_axis = Arrow(start=[0, 0, 0], end=[0, 3.5, 0], color=fg_color, buff=0).shift(
            [-4, 0, 0]
        )
        z_axis = Arrow(start=[0, 0, 0], end=[9, 0, 0], color=fg_color, buff=0).shift(
            [-4, 0, 0]
        )

        c1_text = MathTex(r"\mathbf{C}", color=fg_color).next_to(c1, LEFT)
        ya_text = (
            MathTex("y_w", color=fg_color).next_to(y_axis.end, RIGHT).shift([-4, 0, 0])
        )
        xa_text = (
            MathTex("x_w", color=fg_color).next_to(x_axis.end, UP).shift([-4, 0, 0])
        )
        za_text = (
            MathTex("z_w", color=fg_color).next_to(z_axis.end, RIGHT).shift([-4, 0, 0])
        )

        # self.play(Create(x_axis), Create(y_axis), Create(z_axis))
        # self.play(Write(xa_text), Write(ya_text), Write(za_text))
        self.add(x_axis, y_axis, z_axis, xa_text, ya_text, za_text)
        self.wait(1)
        self.play(Create(c1), Write(c1_text))
        self.wait(2)

        img_plane = Rectangle(
            height=3, width=4, color=fg_color, fill_opacity=1.0, fill_color=bg_color
        )
        matrix = [[1, 0], [1, -1]]
        img_plane.apply_matrix(matrix)
        img_plane.move_to([0, 0, 0])

        c2 = Circle(radius=0.1, color=fg_color, fill_opacity=1)
        c2.move_to(img_plane.get_center())
        p_text = MathTex(r"\mathbf{p}", color=fg_color).next_to(c2, DR)
        za_add = Line(
            start=img_plane.get_center(),
            end=z_axis.end + [-4.5, 0, 0],
            color=fg_color,
            stroke_width=x_axis.stroke_width,
        )
        dashed = DashedLine(
            start=img_plane.get_center(),
            end=[-1.5, 0, 0],
            color=fg_color,
            stroke_width=z_axis.stroke_width,
        )
        plane_text = MathTex(r"\pi", color=fg_color).next_to(img_plane, DL, buff=0.05)
        self.play(Create(img_plane), Create(za_add), Create(dashed), Write(plane_text))
        # self.add(za_add, dashed)
        self.wait(1)
        self.play(Create(c2), Write(p_text))
        self.wait(1)
        end = np.array([1.0, 1, 0])
        end /= np.linalg.norm(end)
        xi_axis = Arrow(
            start=[0, 0, 0],
            end=end,
            color=fg_color,
            buff=0,
            stroke_width=1,
            max_stroke_width_to_length_ratio=10,
            max_tip_length_to_length_ratio=0.1,
        )
        yi_axis = Arrow(
            start=[0, 0, 0],
            end=[0, 1.0, 0],
            color=fg_color,
            buff=0,
            stroke_width=1,
            max_stroke_width_to_length_ratio=10,
            max_tip_length_to_length_ratio=0.1,
        )
        xi_text = MathTex("x_i", color=fg_color).next_to(xi_axis.end, RIGHT, buff=0.1)
        yi_text = MathTex("y_i", color=fg_color).next_to(yi_axis.end, RIGHT, buff=0.1)

        self.play(Create(xi_axis), Create(yi_axis), Write(xi_text), Write(yi_text))

        point = Dot([4, 2.7, 0], color=ray_color)
        point_text = MathTex(r"\mathbf{B}", color=ray_color).next_to(point, RIGHT)
        s = c1.get_center()
        g = point.get_center()
        point_line = DashedLine(s, g, color=ray_color)
        d = g - s
        d /= np.linalg.norm(d)

        inter = s + 5 * d

        intersection = Cross(Dot(inter), stroke_color=ray_color, scale_factor=1.5)
        intersection = Square(0.1, color=ray_color).apply_matrix(matrix).move_to(inter)

        i_text = MathTex(r"\mathbf{b}", color=ray_color).next_to(
            intersection, UR, buff=0.1
        )
        self.play(Create(point), Write(point_text))
        self.wait(1)
        self.play(Create(point_line))
        # self.wait(1)
        self.play(Create(intersection), Write(i_text))

        self.wait(2)

        # self.wait(1)


class CameraTrinagulation(Scene):
    def construct(self):
        # TODO: Add the C1 and C2 points with p1 and p2 descriptions

        self.camera.background_color = bg_color
        print(self.camera.frame_height, self.camera.frame_width)

        self.camera.frame_height *= 1.5
        self.camera.frame_width *= 1.5

        img_plane_1 = Rectangle(height=3, width=4, color=fg_color, fill_opacity=0.0)
        matrix = [[1, 0], [-1, -1]]
        img_plane_1.apply_matrix(matrix)
        img_plane_1.move_to([-5, 0, 0])
        plane_1_text = MathTex(r"\pi_1", color=fg_color).next_to(img_plane_1, LEFT)

        center = img_plane_1.get_center()
        center_circle = Circle(radius=0.05, color=fg_color, fill_opacity=1)
        center_circle.move_to(center)
        p1_text = MathTex(r"\mathbf{p}_1", color=fg_color).next_to(center_circle, LEFT)

        c1 = Circle(radius=0.1, color=fg_color, fill_opacity=1)
        c1.move_to(center - np.array([2, 2, 0]))
        c1_text = MathTex(r"\mathbf{C}_1", color=fg_color).next_to(c1, LEFT)

        line = DashedLine(center - [2, 2, 0], center + [5.2, 5.2, 0], color=CTU_GRAY)
        line.set_stroke(width=1)

        img_plane_2 = Rectangle(height=3, width=4, color=fg_color, fill_opacity=0.0)
        matrix = [[1, 0], [1, 1]]
        img_plane_2.apply_matrix(matrix)
        img_plane_2.move_to([5, 0, 0])
        plane_2_text = MathTex(r"\pi_2", color=fg_color).next_to(img_plane_2, RIGHT)

        center2 = img_plane_2.get_center()
        center_circle2 = Circle(radius=0.1, color=fg_color, fill_opacity=1)
        center_circle2.move_to(center2)
        p2_text = MathTex(r"\mathbf{p}_2", color=fg_color).next_to(
            center_circle2, RIGHT
        )

        c2 = Circle(radius=0.1, color=fg_color, fill_opacity=1)
        c2.move_to(center2 + np.array([2, -2, 0]))
        c2_text = MathTex(r"\mathbf{C}_2", color=fg_color).next_to(c2, RIGHT)

        line2 = DashedLine(
            center2 + np.array([2, -2, 0]), center2 + [-5.2, 5.2, 0], color=CTU_GRAY
        )
        line2.set_stroke(width=1)

        intersect = line_intersection(
            line.get_start_and_end(),
            line2.get_start_and_end(),
        )
        # intersect = line_intersection(line, line2)
        intersect_dot = Dot(intersect, color=fg_color)
        cross = Cross(intersect_dot, stroke_color=CTU_GRAY)
        text = MathTex("P", color=CTU_GRAY)
        text.next_to(intersect_dot, DOWN)

        self.add(
            img_plane_1,
            plane_1_text,
            img_plane_2,
            plane_2_text,
            c1,
            c1_text,
            c2,
            c2_text,
            center_circle,
            p1_text,
            center_circle2,
            p2_text,
        )
        self.wait()

        arrow = CurvedArrow(
            start_point=c1.get_center() + np.array([0.5, -0.5, 0]),
            end_point=c2.get_center() + np.array([-0.5, -0.5, 0]),
            angle=TAU / 4,
            color=RICAIP_GOLD,
        )
        arrow_text = MathTex(r"\mathbf{R,\,t}", color=RICAIP_GOLD).next_to(arrow, DOWN)

        self.play(Create(arrow), Write(arrow_text))

        self.wait(2)

        # self.play(Create(center_circle), Create(center_circle2))
        self.play(Create(line), Create(line2))
        self.play(Create(cross), Create(text))
        self.wait()

        # vert = img_plane_1.get_vertices()

        # print(vert, type(vert))

        x_1 = center_circle.get_center() + np.array([1, 0, 0])
        x_2 = center_circle2.get_center() + np.array([-1, -1, 0])

        x_1 = Dot(x_1, color=ray_color)
        x_2 = Dot(x_2, color=ray_color)

        matrix = [[1, 0], [-1, -1]]
        x_1_box = Square(0.1, color=ray_color).apply_matrix(matrix).move_to(x_1)
        matrix = [[1, 0], [1, 1]]
        x_2_box = Square(0.1, color=ray_color).apply_matrix(matrix).move_to(x_2)

        x_1_text = MathTex(r"\mathbf{b_1}", color=ray_color).next_to(x_1_box, DOWN)
        x_2_text = MathTex(r"\mathbf{b_2}", color=ray_color).next_to(x_2_box, DOWN)

        self.play(Create(x_1_box), Write(x_1_text))
        self.play(Create(x_2_box), Write(x_2_text))
 
        x_1_dir = x_1.get_center() - c1.get_center()
        # x_1_dir /= np.linalg.norm(x_1_dir)
        x_2_dir = x_2.get_center() - c2.get_center()
        # x_2_dir /= np.linalg.norm(x_2_dir)

        x_1_line = DashedLine(
            c1.get_center(), c1.get_center() + 1.7 * x_1_dir, color=ray_color
        )
        x_2_line = DashedLine(
            c2.get_center(), c2.get_center() + 3.3 * x_2_dir, color=ray_color
        )

        X_pt = line_intersection(
            x_1_line.get_start_and_end(),
            x_2_line.get_start_and_end(),
        )
        X_dot = Dot(X_pt, color=ray_color)
        X_cross = Cross(X_dot, stroke_color=ray_color)
        X_text = MathTex(r"\mathbf{B}", color=ray_color).next_to(X_dot, DOWN)

        self.play(Create(x_1_line), Create(x_2_line))

        self.play(Create(X_cross), Write(X_text))

        self.wait(2)


class NeRF(Scene):
    """Should plot snene based on equation
    f(x)=3 exp^(-(((x-3)^(2))/(2*0.4)))+2 exp^(-(((x-6)^(2))/(2*0.5)))-0.01 x+0.2
    """

    def construct(self):
        # self.image()
        self.video()

    def image(self):
        self.camera_setup()
        self.show_axis()
        self.plot_function()
        self.plot_object()
        self.plot_ray()

    def camera_setup(self):
        """Setup camera and background"""
        self.camera.background_color = bg_color
        # self.camera.frame_height *= 1.5
        # self.camera.frame_width *= 1.5

    def plot_object(self):
        """Plot the object to interact with the ray"""
        r1 = Rectangle(height=1, width=0.05, color=CTU_BLUE, fill_opacity=1.0)
        r2 = Rectangle(height=1, width=1.5, color=CTU_ORANGE, fill_opacity=1.0)
        r1.move_to([-2, 3, 0])
        r2.move_to([1 + 0.75, 3, 0])
        dot = Dot([-5, 3, 0], color=fg_color)
        dot_text = MathTex(r"\mathbf{p}", color=fg_color).next_to(dot, LEFT)

        self.add(r1, r2, dot, dot_text)

    def show_axis(self):
        # """Show axis with labels"""
        x_start = np.array([-5.7, -2, 0])
        x_end = np.array([6, -2, 0])

        y_start = np.array([-5, -2.5, 0])
        y_end = np.array([-5, 1, 0])

        x_axis = Line(x_start, x_end, color=fg_color)
        y_axis = Line(y_start, y_end, color=fg_color)
        y_label = MathTex(r"\sigma (t)", color=fg_color).next_to(y_end, LEFT)
        x_label = MathTex(r"t", color=fg_color).next_to(x_end, DOWN)

        Origin_text = MathTex(r"\mathbf{O}", color=fg_color).next_to(
            [-5, -2, 0], DL, buff=0.1
        )

        self.add(x_axis, y_axis, y_label, x_label, Origin_text)

    def plot_function(self):
        """Plot the function"""
        m1 = 2.5
        m2 = m1 * 0.7
        mu_1 = -2
        mu_2 = 1
        sigma_1 = 0.5
        sigma_2 = 0.5

        f = FunctionGraph(
            lambda x: m1 * np.exp(-((x - mu_1) ** 2) / (2 * sigma_1))
            + m2 * np.exp(-((x - mu_2) ** 2) / (2 * sigma_2))
            - 0.01 * x
            + 0.2
            - 2,
            color=ray_color,
            x_range=[-5, 6],
            # strokestroke_color=[PURE_GREEN, WHITE],
        )

        # self.add(f)

        def function(x, m2):
            if x < 1:
                return (
                    m1 * np.exp(-((x - mu_1) ** 2) / (2 * sigma_1))
                    + m2 * np.exp(-((x - mu_2) ** 2) / (2 * sigma_2))
                    - 0.01 * x
                    + 0.2
                    - 2
                )
            if 1 <= x <= 2.5:
                return m2 - 2 + 0.2 - 0.01 * x
            else:
                return (
                    m1 * np.exp(-((x - mu_1) ** 2) / (2 * sigma_1))
                    + m2 * np.exp(-((x - mu_2 - (1 + 0.5)) ** 2) / (2 * sigma_2))
                    - 0.01 * x
                    + 0.2
                    - 2
                )

        dx = 0.01
        x = -5
        prev = function(x, m2)
        gradient_color = fg_color
        while x < 6:
            gradient_color = self.color_based_on_position(x)
            y = function(x, m2)
            # TODO: add color gradient
            line = Line([x, prev, 0], [x + dx, y, 0], color=gradient_color)
            self.add(line)
            prev = y
            x += dx

    def color_based_on_position(
        self,
        x,
    ):
        if x < -4:
            gradient_color = fg_color
        elif -4 <= x < -2:
            gradient = (x + 4) / 2
            gradient_color = interpolate_color(
                ManimColor(fg_color), ManimColor(CTU_BLUE), gradient
            )
        elif -2 <= x < 1:
            gradient = (x + 2) / 3
            gradient_color = interpolate_color(
                ManimColor(CTU_BLUE), ManimColor(CTU_ORANGE), gradient
            )
        elif 1 <= x <= 2.5 + 0.75 / 2 - 0.1:
            gradient_color = CTU_ORANGE

        elif 2.5 <= x <= 2.5 + 2:
            gradient = (x - 2.5) / 2
            gradient_color = interpolate_color(
                ManimColor(CTU_ORANGE), ManimColor(fg_color), gradient
            )
        else:
            gradient_color = fg_color

        return gradient_color

    def plot_ray(self):
        """Plot the ray"""

        dot = Dot([-5, 3, 0], color=fg_color)
        line = Line(dot.get_center(), [6, 3, 0], color=fg_color)
        # line.set_color_by_gradient(CTU_BLUE, CTU_RED)
        # print(line.get_gradient_start_and_end_points())
        self.add(dot, line)

    def move_plot(self):
        """Shows the ray moving in space and plots the density function"""
        start_dot = Dot([-5, 3, 0], color=fg_color)
        dot = Dot(
            [-5, 3, 0], color=fg_color, radius=0
        )  # Invisible dot that is just propagated
        self.curr_x = -5
        self.prev_x = -5

        self.density_curve = VGroup()

        m1 = 2.5
        m2 = m1 * 0.7
        mu_1 = -2
        mu_2 = 1
        sigma_1 = 0.5
        sigma_2 = 0.5

        def density_function(x):
            if x < 1:
                return (
                    m1 * np.exp(-((x - mu_1) ** 2) / (2 * sigma_1))
                    + m2 * np.exp(-((x - mu_2) ** 2) / (2 * sigma_2))
                    - 0.01 * x
                    + 0.2
                    - 2
                )
            if 1 <= x <= 2.5:
                return m2 - 2 + 0.2 - 0.01 * x
            else:
                return (
                    m1 * np.exp(-((x - mu_1) ** 2) / (2 * sigma_1))
                    + m2 * np.exp(-((x - mu_2 - (1 + 0.5)) ** 2) / (2 * sigma_2))
                    - 0.01 * x
                    + 0.2
                    - 2
                )

        def get_density_curve():
            """Updating the density curve"""
            color = self.color_based_on_position(self.curr_x)
            start = [self.prev_x, density_function(self.prev_x), 0]
            end = [self.curr_x, density_function(self.curr_x), 0]
            self.density_curve.add(Line(start, end, color=color))
            return self.density_curve

        def get_ray():
            color = self.color_based_on_position(self.curr_x)
            start = [-5, 3, 0]
            end = [self.curr_x, 3, 0]
            return Line(start, end, color=fg_color)

        def propagate_endpoint(mob, dt):
            # print(dt)
            self.prev_x = self.curr_x
            self.curr_x += 2 * dt if self.curr_x < 6 else 6

            mob.move_to([self.curr_x, 3, 0])

        dot.add_updater(
            propagate_endpoint,
        )

        ray_line = always_redraw(get_ray)
        density_curve = always_redraw(get_density_curve)

        self.add(dot)
        self.add(start_dot, ray_line, density_curve)

        self.wait(5.5)

        dot.remove_updater(propagate_endpoint)
        # ray_line.clear_updaters()
        # density_curve.clear_updaters()

        # TODO: Implement this functioon to plot both the rau and the density function

    def video(self):
        self.camera_setup()
        self.show_axis()
        self.plot_object()
        self.wait(1)

        self.move_plot()

        # self.wait(1)

        # self.move_plot()
        # self.wait(2)
