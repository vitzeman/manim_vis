from manim import *


class CreateCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set the color and transparency
        self.play(Create(circle))  # show the circle on screen


class SquareToCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set color and transparency

        square = Square()  # create a square
        square.rotate(PI / 4)  # rotate a certain amount

        self.play(Create(square))  # animate the creation of the square
        self.play(Transform(square, circle))  # interpolate the square into the circle
        self.play(FadeOut(square))  # fade out animation


class MovingFrameBox(Scene):
    def construct(self):
        text = MathTex(
            "\\frac{d}{dx}f(x)g(x)=",
            "f(x)\\frac{d}{dx}g(x)",
            "+",
            "g(x)\\frac{d}{dx}f(x)",
        )
        self.play(Write(text))
        framebox1 = SurroundingRectangle(text[1], buff=0.1, color=RED)
        framebox2 = SurroundingRectangle(text[3], buff=0.1, color=GREEN)
        self.play(
            Create(framebox1),
        )
        self.wait()
        self.play(
            ReplacementTransform(framebox1, framebox2),
            ReplacementTransform(framebox2, framebox1),
        )
        self.wait()


class SinAndCosFunctionPlot(Scene):
    def construct(self):
        self.camera.background_color = BLACK
        axes = Axes(
            x_range=[-10, 10.3, 1],
            y_range=[-1.5, 1.5, 1],
            x_length=10,
            axis_config={"color": GRAY},
            x_axis_config={
                "numbers_to_include": np.arange(-10, 10.01, 2),
                "numbers_with_elongated_ticks": np.arange(-10, 10.01, 2),
                "decimal_number_config": {"color": GRAY},
                "color": GRAY,
            },
            tips=False,
        )
        axes_labels = axes.get_axis_labels()
        axes_labels.set_color(YELLOW)
        sin_graph = axes.plot(lambda x: np.sin(x), color=BLUE)
        cos_graph = axes.plot(lambda x: np.cos(x), color=RED)
        _2sin_graph = axes.plot(lambda x: 2 * np.sin(x), color=GOLD)

        sin_label = axes.get_graph_label(
            sin_graph, "\\sin(x)", x_val=-10, direction=UP / 2
        )
        cos_label = axes.get_graph_label(cos_graph, label="\\cos(x)")

        _2sin_label = axes.get_graph_label(
            _2sin_graph, "2\\sin(x)", x_val=-10, direction=UP / 2
        )

        vert_line = axes.get_vertical_line(
            axes.i2gp(TAU, cos_graph), color=YELLOW, line_func=Line
        )
        line_label = axes.get_graph_label(
            cos_graph, "x=2\pi", x_val=TAU, direction=UR, color=BLACK
        )

        hor_line = axes.get_horizontal_line(
            axes.i2gp(0, cos_graph), color=YELLOW, line_func=Line
        )
        hor_line.color = YELLOW

        plot = VGroup(axes, sin_graph, cos_graph, vert_line, _2sin_graph, hor_line)
        labels = VGroup(axes_labels, sin_label, cos_label, line_label)
        self.add(plot, labels)


class SineCurveUnitCircle(Scene):
    # contributed by heejin_park, https://infograph.tistory.com/230
    def construct(self):
        self.show_axis()
        self.show_circle()
        self.move_dot_and_draw_curve()
        self.wait()

    def show_axis(self):
        x_start = np.array([-6, 0, 0])
        x_end = np.array([6, 0, 0])

        y_start = np.array([-4, -2, 0])
        y_end = np.array([-4, 2, 0])

        x_axis = Line(x_start, x_end)
        y_axis = Line(y_start, y_end)

        self.add(x_axis, y_axis)
        self.add_x_labels()

        self.origin_point = np.array([-4, 0, 0])
        self.curve_start = np.array([-3, 0, 0])

    def add_x_labels(self):
        x_labels = [
            MathTex("\pi"),
            MathTex("2 \pi"),
            MathTex("3 \pi"),
            MathTex("4 \pi"),
        ]

        for i in range(len(x_labels)):
            x_labels[i].next_to(np.array([-1 + 2 * i, 0, 0]), DOWN)
            self.add(x_labels[i])

    def show_circle(self):
        circle = Circle(radius=3)
        circle.move_to(self.origin_point)
        self.add(circle)
        self.circle = circle

    def move_dot_and_draw_curve(self):
        orbit = self.circle
        origin_point = self.origin_point

        dot = Dot(radius=0.08, color=YELLOW)
        dot.move_to(orbit.point_from_proportion(0))
        self.t_offset = 0
        rate = 0.25

        def go_around_circle(mob, dt):
            self.t_offset += dt * rate
            # print(self.t_offset)
            mob.move_to(orbit.point_from_proportion(self.t_offset % 1))

        def get_line_to_circle():
            return Line(origin_point, dot.get_center(), color=BLUE)

        def get_line_to_curve():
            x = self.curve_start[0] + self.t_offset * 4
            y = dot.get_center()[1]
            return Line(
                dot.get_center(), np.array([x, y, 0]), color=YELLOW_A, stroke_width=2
            )

        self.curve = VGroup()
        self.curve.add(Line(self.curve_start, self.curve_start))

        def get_curve():
            last_line = self.curve[-1]
            x = self.curve_start[0] + self.t_offset * 4
            y = dot.get_center()[1]
            new_line = Line(last_line.get_end(), np.array([x, y, 0]), color=YELLOW_D)
            self.curve.add(new_line)

            return self.curve

        dot.add_updater(go_around_circle)

        origin_to_circle_line = always_redraw(get_line_to_circle)
        dot_to_curve_line = always_redraw(get_line_to_curve)
        sine_curve_line = always_redraw(get_curve)

        self.add(dot)
        self.add(orbit, origin_to_circle_line, dot_to_curve_line, sine_curve_line)
        self.wait(8.5)

        dot.remove_updater(go_around_circle)


class CameraTrinagulation(Scene):
    def construct(self):
        self.camera.background_color = BLACK
        img_plane_1 = Rectangle(height=3, width=4, color=WHITE, fill_opacity=0.0)
        matrix = [[1, 0], [-1, -1]]
        img_plane_1.apply_matrix(matrix)
        img_plane_1.move_to([-3, 0, 0])

        center = img_plane_1.get_center()
        center_circle = Circle(radius=0.1, color=RED, fill_opacity=1)
        center_circle.move_to(center)

        line = DashedLine(center - [2, 2, 0], center + [3.5, 3.5, 0], color=GRAY)
        line.set_stroke(width=1)

        img_plane_2 = Rectangle(height=3, width=4, color=WHITE, fill_opacity=0.0)
        matrix = [[1, 0], [1, 1]]
        img_plane_2.apply_matrix(matrix)
        img_plane_2.move_to([3, 0, 0])

        center2 = img_plane_2.get_center()
        center_circle2 = Circle(radius=0.1, color=RED, fill_opacity=1)
        center_circle2.move_to(center2)

        line2 = DashedLine(
            center2 + np.array([2, -2, 0]), center2 + [-3.5, 3.5, 0], color=GRAY
        )
        line2.set_stroke(width=1)

        intersect = line_intersection(
            line.get_start_and_end(), line2.get_start_and_end()
        )
        # intersect = line_intersection(line, line2)
        intersect_dot = Dot(intersect, color=WHITE)
        cross = Cross(intersect_dot, color=WHITE)
        text = MathTex("P", color=WHITE)
        text.next_to(intersect_dot, DOWN)

        self.add(
            img_plane_1,
            # center_circle,
            img_plane_2,
            # line,
            # center_circle2,
            # line2,
            # intersect_dot,
        )
        self.wait()
        self.play(Create(center_circle), Create(center_circle2))
        self.play(Create(line), Create(line2))
        self.play(Create(cross), Create(text))
        self.wait()

        # self.wait()
