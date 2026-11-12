use std::fs::File ;
use memmap2::Mmap;
use std::collections::HashMap;
// //now handling different format like this
//    Type ID                                           │ GGUF Type                                         │ Rust Type
//   ───────────────────────────────────────────────────┼───────────────────────────────────────────────────┼──────────────────────────────────────────────────
//    0                                                 │ UINT8                                             │ u8 (1 byte)
//    1                                                 │ INT8                                              │ i8 (1 byte)
//    2                                                 │ UINT16                                            │ u16 (2 bytes)
//    3                                                 │ INT16                                             │ i16 (2 bytes)
//    4                                                 │ UINT32                                            │ u32 (4 bytes)
//    5                                                 │ INT32                                             │ i32 (4 bytes)
//    6                                                 │ FLOAT32                                           │ f32 (4 bytes)
//    7                                                 │ BOOL                                              │ bool (1 byte: 0 = false, 1 = true)
//    8                                                 │ STRING                                            │ &'a str (length + UTF-8 bytes)
//    9                                                 │ ARRAY                                             │ item_type (u32) + count (u64) + array items
//    10                                                │ UINT64                                            │ u64 (8 bytes)
//    11                                                │ INT64                                             │ i64 (8 bytes)
//    12                                                │ FLOAT64                                           │ f64 (8 bytes)
#[derive(Debug)]
enum GgufValue<'a> {
        U8(u8),
        I8(i8),
        U16(u16),
        I16(i16),
        U32(u32),
        I32(i32),
        F32(f32),
        Bool(bool),
        String(&'a str),
        Array(Vec<GgufValue<'a>>),
        U64(u64),
        I64(i64),
        F64(f64),
    }

 //struct tensor info for tensor metdata
 #[derive(Debug)] 
 struct TensorInfo<'a>{
    name:&'a str,
    n_dims:u32,
    shape:Vec<u64>,
    tensor_type:u32,
    offset:u64,
 }  
// using the  the mmap i mean borrwing the thing actually 

struct ByteReader <'a>{
    data : &'a [u8],
    offset: usize,
}
//contructor 
impl <'a> ByteReader<'a>{
fn new(data: &'a [u8])->Self{
    Self{data,offset:0}
}
//reading the bytes out of it 
fn read_bytes(&mut self, n:usize) -> &'a [u8]{
   let slices=&self.data[self.offset .. self.offset+n];
    self.offset +=n;
    slices 
}
//reading 32 byte converting the bytes to real numbers the numerical representation 

fn read_u32(&mut self) -> u32{
 let bytes=self.read_bytes(4);
 u32::from_le_bytes(bytes.try_into().unwrap())

}
//reading 64 
fn read_u64(&mut self) ->u64{
    let bytes= self.read_bytes(8);
    u64::from_le_bytes(bytes.try_into().unwrap())
}
fn read_string(&mut self) ->&'a str{
    let len=self.read_u64() as usize;
    let bytes=self.read_bytes(len);
    std::str::from_utf8(bytes).unwrap()
}
fn read_value(&mut self, value_type: u32)-> GgufValue<'a>{
    match value_type{
    0=>{ let byte =self.read_bytes(1)[0];
        GgufValue::U8(byte)
    }
    1=>{ let byte =self.read_bytes(1)[0] as i8;
        GgufValue::I8(byte)
    }
    2=>{
        let bytes =self.read_bytes(2);
        GgufValue::U16(u16::from_le_bytes(bytes.try_into().unwrap()))
    }
    3=>{
        
        let bytes =self.read_bytes(2);
        GgufValue::I16(i16::from_le_bytes(bytes.try_into().unwrap()))
    }
    4=>{ GgufValue::U32(self.read_u32()) //trying this short one got this from a suggestion
    }
    5=>{GgufValue::I32(self.read_u32() as i32)}
    6=>{ let bytes=self.read_bytes(4);
     GgufValue::F32(f32::from_le_bytes(bytes.try_into().unwrap()))}
    7=>{
        let byte =self.read_bytes(1)[0];
        GgufValue::Bool(byte !=0)

    }
    8=>{
        GgufValue::String(self.read_string())
    }
    9=>{
        let item_type=self.read_u32(); //reading the item type 
        let count=self.read_u64() as usize; // reading how many items are there 
        let mut items= Vec::with_capacity(count); // pre allocating the memory so it doesnot need to reallocate repeatedly
        for _ in 0..count{ //loop and use recursion
            items.push(self.read_value(item_type));
                }
        GgufValue::Array(items) // returing the array variant  
    }
   10=>{
    GgufValue::U64(self.read_u64())
   }
   11=>{
    GgufValue::I64(self.read_u64()as i64)
   }
   12=>{
    let bytes=self.read_bytes(8);
    GgufValue::F64(f64::from_le_bytes(bytes.try_into().unwrap()))
   }
   _=> panic!("Unkown value type:{}",value_type),
}}
}
fn main() -> Result<(),Box<dyn std::error::Error>>{
    // model path: first CLI argument, or models/<default>.gguf inside this repo
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "models/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf".to_string()
    });
    let file =File::open(&path)?;
    let mmap=unsafe {Mmap::map(&file)?};
    let mut reader=ByteReader::new(&mmap);
    //checking if this is a GGUF format or not 
    //let magic = &mmap[0..4];
    let magic= reader.read_bytes(4);
    if magic !=b"GGUF"{
        panic!("Not a GGUF file! Found {:?}",magic);
    }
    //Version (4 to 8, 4 bytes as u32):
   //let version =u32::from_le_bytes(mmap [4..8].try_into()?); header ;
   let version=reader.read_u32();
    //Tensorcount (8 to 16, 8 bytes as u64):
    let tensor_count = reader.read_u64();
    //Metadata KV count (16 to 24, 8 bytes as u64):
    let metadata_kv_count = reader.read_u64();
    println!("GGUF Version: {}", version);
    println!("Tensor count: {}", tensor_count);
    println!("Metadata KV count: {}", metadata_kv_count);
    //let reads=ByteReader::new(&mmap);
    //println!("{}",reads.offset);

    // this loop is going to read the general.arhictecture its type is string val-> qwen3.5
   // COMMENTING THIS FIRST LOOP ITS CAUSING THE PANIC ON THE THREAD 
    // for i in 0..metadata_kv_count{
    //     let key= reader.read_string();
    //     let val_type=reader.read_u32();
    //     println!("{}:key={}, type={}",i,key,val_type);

    //     // handling the reading next val
    //     break; // breaking after 1 for now
    // }
    //introducing the difefrent variable which i have created on the code  
    //this is will be reading the meta data 
    //
    for i in 0..metadata_kv_count{
        let key=reader.read_string(); // giving the qwen3.5 the model name 
        //println!("{} the string",key);
        let val_type=reader.read_u32();
        //println!("{}",val_type); // value type is 12 
        let val =reader.read_value(val_type);

        match &val{
            GgufValue::Array(arr)=> println!("{}: {}= Array of {} items",i,key,arr.len()),
            _=> println!("{}:{} = {:?}",i,key,val)
        }
    }
    let mut tensors=HashMap::new();
    for i in 0..tensor_count{
        let name =reader.read_string(); // starting with the tensor name 
        let n_dims=reader.read_u32(); // for handling the dimensions 
        let mut shape= Vec::with_capacity(n_dims as usize); // reading the shape 

        for j in 0..n_dims{
            shape.push(reader.read_u64());
        }
        let tensor_type=reader.read_u32();
        let offset=reader.read_u64();
        //testing it with the first 5

        if i<5{
            println!("{}:'{}' shape={:?} type={} offset={}",i,name,shape,tensor_type,offset);
        }
        tensors.insert(name, TensorInfo{
            name,
            n_dims,
            shape,
            tensor_type,
            offset
        });
         
    }
   let tensor_data_offset=(reader.offset+31)& !31; // --> have made this to round off the offset to the nearest 32 -byte mutiple i have picked this trick of the paper (bitwise  alignment )
   println!("Successfull loaded {} tensor into map!",tensors.len());
   println!("Raw tensor data start at byte:{}",tensor_data_offset);

    Ok(())

}
