/*	
	TRABALHO DE INTELIGÊNCIA ARTIFICIAL APLICADA

	DESENVOLVIMENTO DE UM NEURÔNIO DE ROSENBLATT 
  
*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <conio.h>

// Quantidade de RUs a serem usados no treinamento.
// Foi usada uma função que gera números positivos
// aleatórios a serem usados no treinamento e teste 
// do neurônio.
#define AMOSTRAS 2000

// Quantidade de vezes que o total de amostras 
// é usado para treinar o neurônio
#define EPOCHS 20000

// Taxa de aprendizagem do neurônio.
#define RATE 0.01


// Quantidade de entradas do nosso neurônio.
#define INPUTS 7

// Peso do RU correto no espaço de amostras
// Esse valor vai definir a quantidade mínima
// de ocorrêncis do RU esperado no espaço total
// de amostras.
#define REFORCE 5

// Pesos das entradas + valor de bias
float w[INPUTS+1];


// Matriz com os dados de entrada que são gerados randomicamente pela 
// função make_data()
int input[AMOSTRAS][INPUTS];
// Vetor com os dados de saída indicando se o dado de entrada correspondente
// é o RU do aluno desejado (1) ou não (-1).
int output[AMOSTRAS];


//Inicializa os pesos com valores aleatórios entre -1 e 1 e o bias com -1.
int initialize(float w[]) {
	// Seed para o rand()
	srand(time(NULL));
	for (int i = 0; i < INPUTS; i++)
		w[i] =  2 * ((float)rand() / (float)(RAND_MAX)) - 1;

	// Bias
	w[INPUTS] = -1;
	return(0);
}

// Verifica se um RU gerado é igual ao meu RU.
int ver_ru(int ru[]) {
	// RU do aluno Marcelo Teixeira Campos
	int my[INPUTS] = { 3,3,9,8,7,5,1 };
	for (int i = 0; i < INPUTS; i++)
		if (my[i] != ru[i])
			return(-1);
	return(1);
}

// Cria as amostras e saídas para treinamento.
int make_data() {
	// Cria RUs aleatórios.

	// Seed para que a sequência aleatória não se repita em testes consecutivos.
	srand(time(NULL) + input[0][0] + input[0][1] + input[0][2]);

	// Preenche a Matriz de entrada.
	for (int i = 0; i < AMOSTRAS; i++) {
		for (int j = 0; j < INPUTS; j++) {
			// Peenche de modo randômico as amostras.
			input[i][j] = rand() % 10;
		}
		// Calcula a saída esperada para a respectiva entrada.
		output[i] = ver_ru(input[i]);
	}
	// Garante uma quantidade mínima (REFORCE) de RUs válidos (meu RU) no espaço de amostras.
	for (int i = 0; i < REFORCE; i++) {
		int pos = rand() % AMOSTRAS;
		int my[INPUTS] = { 3,3,9,8,7,5,1 };
		for (int i = 0; i < INPUTS; i++)
			input[pos][i] = my[i];
		output[pos] = 1;
	}
	return(0);
}

// Calcula a saída do neurônio
int predict(int x[]) {
	float sum = 0;
	// Somatório das entradas multiplicadas pelos pesos.
	for (int i = 0; i < INPUTS; i++)
		sum += x[i] * w[i];
	// Bias.
	sum += w[INPUTS];
	// Função de ativação degrau unitário.
	if (sum > 0) return(1);
	return(-1);
}

float neuron(int x[]) {
	float sum = 0;
	// Somatório das entradas multiplicadas pelos pesos.
	for (int i = 0; i < INPUTS; i++)
		sum += x[i] * w[i];
	// Bias.
	sum += w[INPUTS];
	
	
	return(sum);
}

// Treina o neurônio pela regra delta e retorna o erro global
float train(int x[AMOSTRAS][INPUTS], int output[], float rate) {
	float w_delta[INPUTS] = { 0,0,0,0,0,0,0 };

	double global_delta = 0;

	// Erro por amostra.
	int erro = 0;

	// Calcula o Delta para cada entrada.
	for (int j = 0; j < AMOSTRAS; j++) {
		// Calcula o erro da amostra j.
		erro = output[j] - neuron(x[j]);
		for (int i = 0; i < INPUTS; i++) {
			w_delta[i] += rate * erro * x[j][i];
		}
		global_delta += pow(erro,2);
	}

	//printf("Global Delta: %.2f\n", global_delta);

	// Atualiza os pesos.
	for (int i = 0; i < INPUTS; i++)
		w[i] += w_delta[i] / AMOSTRAS;

	return(global_delta/2);
}


// Testa o neurônio.
int teste() {
	int erros = 0;
	int falso_pos = 0;
	int falso_neg = 0;
	
		// Verifica a predição do neurônio e registra os falsos positivos e falsos negativos.
		for (int i = 0; i < AMOSTRAS; i++) {
			if (output[i] > predict(input[i]))
				falso_neg++;
			if (output[i] < predict(input[i]))
				falso_pos++;
			if (output[i] != predict(input[i]))
				erros++;
		}
	
	// Imprime na tela o relatório do teste.
	printf("TOTAL: %d Amostras - ERRO: %.2f%%  falsos positivos: %d falsos negativos: %d\n", AMOSTRAS ,100*(float)erros / (float)( AMOSTRAS), falso_pos, falso_neg);
	return(0);
}

int main() {

	// Learning rate, taxa de aprendizagem.
	float l_rate = RATE;
	// Épocas de treinamento. Qauntidade de vezes que o conjunto 
	// de amostras vai ser treinado no neurônio.
	int epochs = EPOCHS;

	// Erro global.
	float mse ;

	// Cria os dados de input e output.
	make_data();
	// Inicializa randomicamente os pesos e o bias.
	initialize(w);

	// Contador de épocas
	
	#define GCOUNT 30
	float d_epoch = epochs / GCOUNT;
	float c_epoch = 0;

	for (int j = 1; j <= epochs; j++) {
		// Output gráfico.
		if (j >= c_epoch * d_epoch) {
			system("cls");
			printf("Treinando %d epochs\n",epochs);
			printf("[");
			for(int k = 0; k < c_epoch; k++)
				printf("=");
			for(int k = c_epoch; k < GCOUNT; k++)
				printf(" ");
			printf("]\n");
			c_epoch++;
		}
		// Treinamento.
		mse = train(input, output, l_rate);
		if ( mse == 0) {
			printf("MSE 0 atingido em %d épocas!\n", j);
			break;
		}
	}
	printf("Dados de treinamento:\n");
	teste();

	// Cria um novo espaço de amostras.
	printf("Dados de teste:\n");
	make_data();
	teste();

	// Informa os pesos resultantes.
	printf("Pesos obtidos: ");
	for (int i = 0; i < INPUTS; i++)
		printf("%f ", w[i]);

	printf("bias %f ",w[INPUTS]);
	return(0);
}
