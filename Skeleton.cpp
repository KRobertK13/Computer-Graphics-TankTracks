//=============================================================================================
// Mintaprogram: Zold haromszog. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Kovacs Robert Kristof
// Neptun : R92D9T
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
//=============================================================================================
// Virus + Antibody
//=============================================================================================
#include "framework.h"

template<class T> struct Dnum {
	float f;
	T d;
	Dnum(float f0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) { return Dnum(f * r.f, f * r.d + d * r.f); }
	Dnum operator/(Dnum r) { return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f); }
};

template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return  Dnum<T>(sinh(g.f), cosh(g.f) * g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return  Dnum<T>(cosh(g.f), sinh(g.f) * g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }

typedef Dnum<vec2> Dnum2;

struct Camera {
	vec3 wEye, wLookat, wVup;
	float fov, asp, fp, bp;
public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 200;
	}
	mat4 V() {
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}

	mat4 P() {
		return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp * bp / (bp - fp), 0);
	}
};

class SteelTexture : public Texture {
public:
	SteelTexture(const int width, const int height) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 steel(0.44, 0.47, 0.49, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) image[y * width + x] = steel;
		create(width, height, image, GL_NEAREST);
	}
};

class SandTexture : public Texture {
public:
	SandTexture(const int width, const int height) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 sand(0.76, 0.70, 0.5, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) image[y * width + x] = sand;
		create(width, height, image, GL_NEAREST);
	}
};

class BrickTexture : public Texture {
public:
	BrickTexture(const int width, const int height) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 light(0.7176, 0.3294, 0.1765, 1), dark(0.6353, 0.31765, 0.2, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? light : dark;
		}
		create(width, height, image, GL_NEAREST);
	}
};

struct RenderState {
	mat4	    M, Minv, V, P;
	Texture* texture;
	vec3	    wEye;
};

class PhongShader : public GPUProgram {
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		const vec3 wLightPos  = vec3(10, 10, -5);	// directional light source;
		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight;		    // light dir in world space
		out vec2 texcoord;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
		    wView  = wEye - (vec4(vtxPos, 1) * M).xyz;
			wLight = wLightPos;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";
	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		const vec3 ks = vec3(2, 2, 2);
		const float shininess = 50.0f;
		const vec3 La = vec3(0.1f, 0.1f, 0.1f);
		const vec3 Le = vec3(2, 2, 2);    

		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight;        // interpolated world sp illum dir
		in  vec2 texcoord;
		
        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;
			vec3 kd = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = kd * 3.14;
			vec3 L = normalize(wLight);
			vec3 H = normalize(L + V);
			float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
			fragmentColor = vec4(ka * La + (kd * cost + ks * pow(cosd, shininess)) * Le, 1);
		}
	)";
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		setUniform(state.M * state.V * state.P, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
	}
};

PhongShader* shader;

struct VertexData {
	vec3 position, normal;
	vec2 texcoord;
};

class Geometry {
protected:
	unsigned int vao, vbo;
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
	}
	void Load(const std::vector<VertexData>& vtxData) {
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vtxData.size() * sizeof(VertexData), &vtxData[0], GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}
	virtual void Draw() = 0;
	virtual void Animate(float t) { }
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

class ParamSurface : public Geometry {
	unsigned int nVtxPerStrip, nStrips;
public:
	virtual VertexData GenVertexData(float u, float v) = 0;

	void create(int N = 30, int M = 30) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;	
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		Load(vtxData);
	}

	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
};

class Plain : public ParamSurface {
public:
	Plain() { create(); }

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		float phi = u * 100.0f, theta = v * 100.0f;
		vd.normal = vec3(0, 0, 1);
		vd.position = vec3(cosf(theta), 0, sinf(phi)) * 1000;
		vd.texcoord = vec2(u, v);
		return vd;
	}
};

class Plate : public ParamSurface {
public:
	Plate() { create(); }

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		float phi = u, theta = v;
		vd.normal = vec3(0, 0, 1);
		vd.position = vec3(cosf(theta), 0, sinf(phi) * 0.09f);
		vd.texcoord = vec2(u, v);
		return vd;
	}
};

class Tractricoid : public ParamSurface {
public:
	Tractricoid() { create(); }

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		u = u;
		v = v;
		const float height = 5.0f;
		Dnum2 U(v * height, vec2(1, 0)), V(u * 2 * M_PI, vec2(0, 1));
		Dnum2 X = Cos(V) / Cosh(U), Y = Sin(V) / Cosh(U), Z = U + Tanh(U) * (-1);
		vd.position = vec3(X.f, Y.f, Z.f);
		vd.normal = cross(vec3(X.d.x, Y.d.x, Z.d.x), vec3(X.d.y, Y.d.y, Z.d.y));
		vd.texcoord = vec2(u, v);
		return vd;
	}
};

struct Object {
	Texture* texture;
	Geometry* geometry;
	vec3 translation, rotationAxis;
	float rotationAngle = 0;
public:
	Object(Texture* _texture, Geometry* _geometry) :
		translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		texture = _texture;
		geometry = _geometry;
	}
	virtual void Draw(RenderState state) {
		state.M = RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		state.Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis);
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}
	virtual void Animate() {}
};

vec4 qmul(vec4 q1, vec4 q2) {
	vec3 d1(q1.x, q1.y, q1.z), d2(q2.x, q2.y, q2.z);
	vec3 imag = d2 * q1.w + d1 * q2.w + cross(d1, d2);
	return vec4(imag.x, imag.y, imag.z, q1.w * q2.w - dot(d1, d2));
}

mat4 quatToMatrix(vec4 q, vec4 qinv) {
	return mat4(qmul(qmul(q, vec4(1, 0, 0, 0)), qinv),
		qmul(qmul(q, vec4(0, 1, 0, 0)), qinv),
		qmul(qmul(q, vec4(0, 0, 1, 0)), qinv),
		vec4(0, 0, 0, 1));
}

float rnd() {
	int rndE = (int)rand() % 2;
	return rndE ? (float)((int)rand() % 150) : -(float)((int)rand() % 150);
}

struct Chain : Object {
	Chain(Texture* _texture, Geometry* _geometry) : Object(_texture, _geometry) {}
	void Draw(RenderState state) {
		state.M = RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation) * state.M;
		state.Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * state.Minv;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}
};

struct TankTrack : Object {
	std::vector<Chain> chains;
	vec3 pivot = vec3(0, 0.12f, 0.54f);
	float speed = 0.01f;

public:
	TankTrack(Texture* _texture, Geometry* _geometry, float x, float z) : Object(_texture, _geometry) {
		pivot.x = x;
		pivot.z = z;
		vec3 tran = vec3(pivot.x, 0.241f, 0);
		float fu = 1.08f + M_PI * 0.12f;
		for (int i = 0; i < 28; i++) {
			if (tran.z > fu) { tran.z = 1.08f - (tran.z - fu); tran.y = 0.01f; }
			chains.push_back(Chain(texture, geometry));
			chains[i].rotationAxis = vec3(1, 0, 0);
			if (tran.z - pivot.z > 0.54f) {
				chains[i].rotationAngle = (tran.z - 1.08f) / 0.12f;
				chains[i].translation = vec3(pivot.x, 0.12f * sinf((M_PI / 2) - chains[i].rotationAngle) + 0.13f, 0.12f * cosf((M_PI / 2) - chains[i].rotationAngle) + pivot.z + 0.54);
				tran.z += 0.104f;
			}
			else if (pivot.z - tran.z > 0.54f) {
				chains[i].rotationAngle += (-tran.z / 0.12f);
				chains[i].translation = vec3(pivot.x, 0.12f * sinf((3 * M_PI / 2) - chains[i].rotationAngle) + 0.13f, 0.12f * cosf((3 * M_PI / 2) - chains[i].rotationAngle) + pivot.z - 0.54);
				tran.z -= 0.104f;
			}
			else if (tran.y > pivot.y) {
				chains[i].rotationAngle = 0;
				chains[i].translation = vec3(pivot.x, 0.241f, tran.z);
				tran.z += 0.104f;
			}
			else {
				chains[i].rotationAngle = 0;
				chains[i].translation = vec3(pivot.x, 0.001f, tran.z);
				tran.z -= 0.104f;
			}
		}
	}

	void Draw(RenderState state) {
		for (size_t i = 0; i < chains.size(); i++)
		{
			chains[i].Draw(state);
		}
	}

	void speedUp() {
		speed += 0.003f;
	}

	void slowDown() {
		speed -= 0.003f;
	}

	void Animate() {
		for (size_t i = 0; i < 28; i++)
		{
			if (chains[i].translation.z - pivot.z > 0.54f) {
				chains[i].rotationAngle += speed / 0.12f;
				chains[i].rotationAxis = vec3(1, 0, 0);
				chains[i].translation = vec3(chains[i].translation.x, 0.12f * sinf((M_PI / 2) - chains[i].rotationAngle) + 0.13f, 0.12f * cosf((M_PI / 2) - chains[i].rotationAngle) + pivot.z + 0.54);
			}
			else if (pivot.z - chains[i].translation.z > 0.54f) {
				chains[i].rotationAngle += speed / 0.12f;
				chains[i].rotationAxis = vec3(1, 0, 0);
				chains[i].translation = vec3(chains[i].translation.x, 0.12f * sinf((3 * M_PI / 2) - chains[i].rotationAngle) + 0.13f, 0.12f * cosf((3 * M_PI / 2) - chains[i].rotationAngle) + pivot.z - 0.54);
			}
			else if (chains[i].translation.y > pivot.y) {
				chains[i].rotationAngle = 0.0f;
				chains[i].translation = vec3(chains[i].translation.x, 0.241f, chains[i].translation.z + speed);
			}
			else {
				chains[i].rotationAngle = 0.0f;
				chains[i].translation = vec3(chains[i].translation.x, 0.001f, chains[i].translation.z - speed);
			}
		}
	}
};

struct Tank : Object {
	TankTrack* leftTrack;
	TankTrack* rightTrack;
	vec3 pivot;
	float speed = 0.01f;
	float width;
	float w;
	Tank(Texture* _texture, Geometry* _geometry) : Object(_texture, _geometry) {
		leftTrack = new TankTrack(_texture, _geometry, 0, 0.54f);
		rightTrack = new TankTrack(_texture, _geometry, -1.5, 0.54f);
		pivot = vec3(0, 0, 0);
		width = 1.5f;
		w = 0;
		rotationAxis = vec3(0, 1, 0);
		rotationAngle = 0;
	}

	float turnL() {
		leftTrack->speed -= 0.003f;
		w = (leftTrack->speed - rightTrack->speed) / width;
		return w;
	}

	float turnR() {
		rightTrack->speed -= 0.003f;
		w = (leftTrack->speed - rightTrack->speed) / width;
		return w;
	}

	void Draw(RenderState state) {
		state.M = RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		state.Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis);

		leftTrack->Draw(state);
		rightTrack->Draw(state);
	}

	void Animate() {
		rotationAngle -= w;
		vec4 v = (vec4(0, 0, 1, 1) * RotationMatrix(rotationAngle, rotationAxis));
		translation = translation + vec3(v.x, v.y, v.z) * speed;
		pivot = translation;
		rightTrack->Animate();
		leftTrack->Animate();
	}

	void speedUp() {
		leftTrack->speedUp();
		rightTrack->speedUp();
		speed += 0.003f;
	}

	void slowDown() {
		leftTrack->slowDown();
		rightTrack->slowDown();
		speed -= 0.003f;
	}

};

class Scene {
	Camera camera;
	Tank* tankObject;
	std::vector<Object*> buildingObject;
	Object* plain;
	float rotation = 0;
	float w = 0;
public:
	void Build() {
		shader = new PhongShader();
		plain = new  Object(new SandTexture(1, 1), new Plain());
		plain->translation = (-500, 0, -500);
		tankObject = new Tank(new SteelTexture(1, 1), new Plate());
		for (size_t i = 0; i < 500; i++)
		{
			Object* tetra = new Object(new BrickTexture(100, 100), new Tractricoid());
			tetra->translation = vec3(rnd(), 0, rnd());
			tetra->rotationAngle = 3 * M_PI / 2;
			tetra->rotationAxis = vec3(1, 0, 0);
			buildingObject.push_back(tetra);
		}
		camera.wEye = vec3(0, 1.5, -1.5); camera.wLookat = vec3(0, 1, 0); camera.wVup = vec3(0, 1, 0);
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V(); state.P = camera.P();
		for (size_t i = 0; i < 500; i++) buildingObject[i]->Draw(state);
		plain->Draw(state);
		tankObject->Draw(state);
	}

	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V(); state.P = camera.P();
		for (size_t i = 0; i < 500; i++) buildingObject[i]->Draw(state);
		tankObject->Draw(state);
		RenderState s;
		s.wEye = vec3(-500, 1, -500);
		s.V = camera.V(); s.P = camera.P();
		plain->Draw(s);
	}

	void Animate(float tstart, float tend) {
		tankObject->Animate();
		rotation += w;
		vec4 v = (vec4(0, 0, 1, 1) * RotationMatrix(rotation, vec3(0, -1, 0)));
		camera.wEye = tankObject->pivot - vec3(v.x, -1.5, v.z);
		camera.wLookat = tankObject->pivot + vec3(0, 1, 0);
	}

	void speedUp() {
		tankObject->speedUp();
	}

	void slowDown() {
		tankObject->slowDown();
	}

	void turnL() {
		w = tankObject->turnL();
	}

	void turnR() {
		w = tankObject->turnR();
	}
};

Scene scene;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}

void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	scene.Render();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	printf("%c\n", key);
	if (key == 'q') scene.speedUp();
	else if (key == 'a') scene.slowDown();
	else if (key == 'i') scene.turnL();
	else if (key == 'o') scene.turnR();
}
void onKeyboardUp(unsigned char key, int pX, int pY) { }
void onMouse(int button, int state, int pX, int pY) { }
void onMouseMotion(int pX, int pY) {}

void onIdle() {
	static float tend = 0, tEvent = 0;
	const float dt = 0.05f, dtEvent = 0.1f;
	float t = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	while (t < tend) {
		float te = fmin(fmin(t + dt, tend), tEvent);
		scene.Animate(t, te);
		if (te == tEvent) {
			tEvent += dtEvent;
		}
		t = te;
	}
	glutPostRedisplay();
}
