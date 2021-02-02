# CHARACTER_LIST = '‘ẹjO:òồọðỡụẽÂưiŨdỜ”ẦVMẩCệ“+v!ỹỉPt2/lứỄỐƠúằíỔẢắDqữ?SJỳẳấwƯ*ẫQơ3éeĩTÔuỲbâựỰÐể0ùừWợBô²èá4êỷÍỖRL,ộ9ynf%g~³¼ẤỗHIG6Ệử°Yậổx(Ởra&Ẩẵễóở)Đm>Áềẻ;sàỮõẬăỵủkỦì]_Ạ7–ÕNãịố8hỪ…\'[Eớ.ạũếXĂờUỒÝỨảpÀầặ1đcýÚ’zÊỏẶoKZ5"FA-ÌỀỘẾỌÙỤÒÓ=Ã'
CHARACTER_LIST = 'ỏEÓ5UÁẠCẶỸỌù3ủỹ“Ỡỵịsộ>iẨaẢƠúýôẤÐàỐBÌẾữá”Ứ–"ỎỜ‘ŨẺÙYừK2Ẽdâ_XÀÕ:íÉ\'ớỢãơỘ0lửÚ1ỤcỳỖỔÂỲAm[JÔậỡăZỞĩêÊNDHFẲỚ]wểẩố8eSpỂệuPự7ẫõỬ69Ẫèẵẹằồ³ề%&ắẮỰẴẳ~ễqÈỀÃẰợ/4W¼,*tỮ?kở!ổ)-nIógảụéðÒỆòầưỪấLQGẬỊũỒỗĂẦặ²vƯVÝ…Ỉy’ọỄẻìạfđẸbMÍờz°ỷ;(+ứ.ỶĐjĨỦohếxỴRỉẽOTr=@'

WORD_TAG = {"B-W": 0, "I-W": 1, "<PAD_WORD>": 2}

POS_LIST = ['R', 'Nc', 'Vy', 'M', 'A', 'Z', 'N', 'CH', 'Ny', 'Nu', 'C', 'V', 'L', 'I', 'P', 'Np', 'T', 'FW', 'X', 'E', '<PAD_POS>']

NUM_POS_TAGS = len(POS_LIST) - 1

POS2INDEX = {pos: i for i, pos in enumerate(POS_LIST)}

CHARACTER2INDEX = {character: i for i, character in enumerate(CHARACTER_LIST)}
CHARACTER2INDEX['<UNK>'] = len(CHARACTER2INDEX)
CHARACTER2INDEX['<PAD>'] = len(CHARACTER2INDEX)

MAX_LEN = 255
MAX_LEN_CHAR = 18