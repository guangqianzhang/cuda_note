#include <stdio.h>
// 乘法对称量化   //非对称量化
void Mul(
    char** input_a,char** inpput_b,char ** output,const unsigned int num_of_elements,
    const float scale_a,const float scale_b,const float scale_c){
            for (unsigned int i =0;i<num_of_elements;i++)
                for (unsigned int j=0;j<num_of_elements;j++)
                    // output[i][j]=clip(round_fn(input_a[i][j]*scale_a*inpput_b[i][j]*scale_b/scale_c)); //scale ab 量化到scale_c
                    // output[i][j]=clip(round_fn(input_a[i][j]*inpput_b[i][j]*(scale_b*scale_a)/scale_c)); // 量化到scale_c
                    //硬件没有浮点运算能力
                    output[i][j]=clip(input_a[i][j]*inpput_b[i][j]<<round_fn(log((scale_b*scale_a)/scale_c))); 

}
//Quantised Add
void Add(
    char** input_a,char** inpput_b,char ** output,const unsigned int num_of_elements,
    const float scale_a,const float scale_b,const float scale_c){
        const float scale_i= scale_a;  // 要求a b一致！！！！
            for (unsigned int i =0;i<num_of_elements;i++)
                for (unsigned int j=0;j<num_of_elements;j++)
                // output[i][j] =(input_a[i][j]*scale_a+inpput_b[i][j]*scale_b)/scale_c;       
                output[i][j] =(input_a[i][j]+inpput_b[i][j]*scale_i/scale_c);       
    }

//Quantized Activation
void Cilp(
    char** intput,char ** output,float min,float max,
    const float in_scale,const float out_scale, const unsigned int num_of_elements,
    const float scale){
            for (unsigned int i =0;i<num_of_elements;i++)
                for (unsigned int j=0;j<num_of_elements;j++)
                {
                    // output[i][j]=MAX(intput[i][j],min);
                    // output[i][j]=MIN(intput[i][j],max);

                // // input 被量化 min max也被量化
                //     output[i][j]=MAX(intput[i][j]*in_scale,min)/out_scale;
                //     output[i][j]=MIN(intput[i][j]*in_scale,max)/out_scale;  

                //要求inscale outscale一致
                    output[i][j]=MAX(intput[i][j],min/scale);
                    output[i][j]=MIN(intput[i][j],max/scale);                                     

                }
    }
/* 
被动量化算子： scale 被输入输出共享，算子的运算不改变量化参数，给相同量化参数
Pad Clip Relu MaxPooling Reshape Concat Split Transpose Slice Permute
量化参数：
 */
// Quantized  Gemm 矩阵乘法
void MatMul(
    ELEMENT_TYPE** intput,ELEMEN_TYPE** weight,ELEMEN_TYPE* bias,
    ELEMEN_TYPE** output,const unsigned int num_of_elements){
        ACCUMULATOR_TYPE Accumulator[16]; //累加器 暂时寄存累加结果
        for (unsigned int i=0;i<num_of_elements;i+=4){
            //send pachedA to L2
            ELEMEN_TYPE* packedA=LhsPackElement(intput,num_of_elements,i);//把数据送上缓存
            for(unsigned int j=0;j<num_of_elements;j+=4){
                //send PackedB to L2
                ELEMEN_TYPE* packedB=RhsPackElement(weight,num_of_elements,j);
                for (unsigned int k=0;k<num_of_elements;k+=1){
                    MatMul4x4(packedA,packedB,Accumulator,k);//分块矩阵乘，8bit项量化加速结果16-32bit
                }
                for(unsigned int k=0;k<4;k+=1){ //Rescale 到int8。bias被动量化，与Accumulator量化参数一致！！！
                    output[i+k][j+0]=Accumulator[0]+bias[i+k];
                    output[i+k][j+1]=Accumulator[0]+bias[i+k];
                    output[i+k][j+2]=Accumulator[0]+bias[i+k];
                    output[i+k][j+3]=Accumulator[0]+bias[i+k];
                }
            }
        }
    }
__declspec(noonline) void MatMul4x4(
    ELEMEN_TYPE* packedA ,ELEMEN_TYPE* packedB,
    ACCUMULATOR_TYPE* accumulator, unsigned int offset){
        //计算仅操作下列元素
        // packedA[offset:offset+16]
        // packedB[offset:offset+16]
        //c[16]共计48个，可以全部放入寄存器与L1Cache
        accumulator[0]=packedA[0]*packedB[0]+accumulator[0];
        accumulator[1]=packedA[0]*packedB[1]+accumulator[1];
        accumulator[2]=packedA[0]*packedB[2]+accumulator[2];
        accumulator[3]=packedA[0]*packedB[3]+accumulator[3];
        //....
    }
/* 
数据什么时候转化，
什么算子共享
    */

